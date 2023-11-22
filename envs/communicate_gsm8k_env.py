import re, string, os, sys
from typing import List, Union, Literal
from enum import Enum
import os.path


import os
import copy
import requests
import ast
import openai
import time
import json
import random
import pickle
import torch
import threading
from tqdm import tqdm

import torch.distributed as dist

from iterative_ppo import PPOTrainer, PPODataset, PPODataCollatorForLanguageModeling, ActorCriticLlama, RolloutBuffer

def create_prompt(user_query):
    system_message = "You are an AI assistant. Provide a detailed answer so user don't need to search outside to understand the answer."
    prompt = "### System: " + system_message + "\n"
    prompt += "### Human: " + user_query #+ "\n"
    # prompt += "### Assistant:"
    return prompt



def gen_from_gpt(content, system_msg=None):
        if system_msg is None:
            system_msg = 'you are a Math teacher who is capable of correct student mistake step by step.'
        openai.api_key  = os.environ.get("OPENAI_API_KEY")
        MAX_API_RETRY = 3
        for i in range(MAX_API_RETRY):
            try:
                response = openai.ChatCompletion.create(
                    engine="gpt-4",
                    messages = [{
                                        'role': 'system',
                                        'content': system_msg,
                                    }, {
                                        'role': 'user',
                                        'content': content,
                                    }],
                    temperature=0.7,
                    max_tokens=512,
                    # max_tokens=256,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None)['choices'][0]['message']['content']
                return response
            except Exception as e:
                print(e)
                time.sleep(5)
        print(f'Failed after {MAX_API_RETRY} retries.')
        return 'error'


def generate_new_math_question(qa_pair):
    if type(qa_pair) == str:
        question = qa_pair
        student_response=None
    else:
        question, student_response = qa_pair
    if student_response is None:
        content = f"Here is an example of math problem: {question} \n generate a new math problem that has a single integer answer, which only requires student to answer with a single integer number.\n Please start directly:"
    else:
        content = f"Here is an example of math problem and the solution of the student: {question} \nThe student's solution:\n{student_response}\n generate a new similar math problem that has a single integer answer, which only requires student to answer with a single integer number.\n Please start directly:"
    system_message = 'you are a Math teacher who is capable of generating a new high quality math problem.'
    new_question = gen_from_gpt(content,system_message, max_tokens=512, temperature=0.7)
    
    return new_question


def generate_math_answer(question):
    content = f"Please give a step by step answer to the question, you have to put your final numeric answer in the end, without any extra sign, prefix and surfix, just pure integer numbers, in the format: \n#### answer\n Done, make sure to separate the final numeric answer with  \n#### "
    system_message = 'you are a Math teacher who is capable of generating high quality math solutions.'
    
    response = gen_from_gpt(question + '\n' + content, system_message)
    return question, response

def extract_answer(completion):
    start_idx = completion.find("####")
    # start_idx = completion.find("\n### Answer:")
    # start_idx = 0
    if start_idx == -1:
        # return None
        return completion, 'None'
    for idx, char in enumerate(completion[start_idx:]):
        if char == "\n":
            break
    res = completion[start_idx:start_idx+idx]
    response = completion[:start_idx+idx]
    answer = res[4:].strip()
    return response, answer


class GSM8kCoordinator:
    def __init__(self,
                 json_data,
                 llm,
                 tokenizer,
                 output_dir,
                 ) -> None:
        
        # self.max_tokens = 256
        self.max_tokens = 512
        self.n_example = 1
        self.global_rank = dist.get_rank()
        self.OUTPUT_DIR = output_dir
        self.ppo_buffers = []
        self.use_gt_solution = False
        self.json_data = json_data
        self.set_llm(llm=llm, tokenizer=tokenizer)
        self.LOG_PATH = os.path.join(self.OUTPUT_DIR, 'log', 'print', f'print_log_rank{self.global_rank}.txt')
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'log', 'print')):
            os.makedirs(os.path.join(self.OUTPUT_DIR, 'log', 'print'), exist_ok = True)
        
        if 'completion' in json_data[0]:
            # self.gpt_feedback = False
            self.gpt_feedback = True
            self.envs = [GSM8kEnv(question=row['prompt'], 
                                    key=row['answer'],
                                    tokenizer=tokenizer,
                                    max_steps=1) for row in json_data] 
            for i, env in enumerate(self.envs):
                env.gt_solution = json_data[i]['completion'] + "\n####" + json_data[i]['answer'] + "\n"
        else:
            # test data
            self.gpt_feedback = False
            self.envs = [GSM8kEnv(question=row['text'], 
                                    key=row['metadata'],
                                    tokenizer=tokenizer,
                                    max_steps=1) for row in json_data] 
    
    def set_llm(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer

    def run(self, envs):
        active_envs = [env for env in envs if not (env.is_finished() or env.is_halted())]
        while active_envs:
        
            queries_assistant = []
            for i, env in enumerate(active_envs):
                queries_assistant.append(env.query_assistant()) 
            outputs_assistant = self.prompt_llm(queries_assistant) # TODO parallel with vllm
            for i, env in enumerate(active_envs):
                text_assistant, tokens, logprobs, state_values = outputs_assistant[i]
                response, answer = extract_answer(text_assistant)
                env.update_scratchpad(response, tokens, logprobs, state_values, mask=0) 
                env.answer = answer
                env.solution = response

                if env.is_correct():
                    env.buffer.rewards[-1] = 1.
                    env.finished = True
                    
                    self.print_log('Assistant:' + response)
                    self.print_log(f'Answer: {env.key}')
                    self.print_log('Answer is CORRECT')
                else: 
                    env.buffer.rewards[-1] = -1.
                    
                    self.print_log('Assistant:' + response)
                    self.print_log(f'Answer: {env.key}')
                    self.print_log('Answer is INCORRECT')
                    
                
                env.step_n += 1
                # if env.is_halted() and not env.is_finished():
                #     env.buffer.rewards[-1] = -1.
                
            active_envs = [env for env in envs if not (env.is_finished() or env.is_halted())]
    


    def process_gsm8k_train(self, sample):
        # data = "### Human: " + sample['prompt'] + "\n"
        # data += "### Assistant:" + sample['completion'] + "\n"
        data = create_prompt(sample['prompt'])
      
        data += "\n### Assistant:"
        # data += sample['completion'] + "\n### Answer:" + sample['answer'] + "\n"
        data += sample['completion'] + "\n####" + sample['answer'] + "\n"
        return data
    
    def get_instruction_finetune_data(self, init_n_train=1024):
        """init train data --yadong
        """
        # assert len(self.json_data) >= init_n_train
        self.success_chat_data = [] 
        self.success_examples = []
        train_chat_data = [self.process_gsm8k_train(sample) for sample in self.json_data[:init_n_train]]
        return train_chat_data
        # self.train_chat_data = self.tokenize(self.train_chat_data)
        # self.train_input_ids = torch.tensor(self.train_chat_data['input_ids']).to(self.model.device)
        # self.train_attention_mask = torch.tensor(self.train_chat_data['attention_mask']).to(self.model.device)
        
    def multiple_processes_gpt(self, tasks, func):
        # TODO check the order of outputs
        outputs = []
        # batch_size = 16
        batch_size = 8
        for i in range(len(tasks)//batch_size + 1):
            batch_questions = tasks[i*batch_size:(i+1)*batch_size]
            if len(batch_questions) == 0:
                break
            threads = []
            results = []
            start_time = time.time()
            for question in batch_questions:
                t = threading.Thread(target=lambda result, question: result.append(func(question)), args=(results, question))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()
            outputs.extend(results)
        return outputs
        
    
    def ppo_trial_generate(self, n_iter, test=True):
        self.llm.eval()
        log = ''
        self.run(self.envs)
        if not test:
            ppo_buffers_0 = [env.buffer for env in self.envs]
            ppo_buffers_1 = []
            ppo_buffers_2 = []
            ppo_buffers_3 = []
            if self.gpt_feedback:
                self.print_log('------GPT starts to generate answer--------')
                questions = []
                for env in self.envs:
                    questions.append(env.question)
                gt_qa_pairs = self.multiple_processes_gpt(questions, generate_math_answer)
                gt_qa_pairs_dict = {gt_qa_pair[0]: gt_qa_pair[1] for gt_qa_pair in gt_qa_pairs}
                for env in self.envs:
                    if gt_qa_pairs_dict[env.question] != 'error':
                        env.gt_solution = gt_qa_pairs_dict[env.question] 
                self.print_log('------GPT starts to generate new data--------')
                qa_pairs = []
                for env in self.envs:
                    for i in range(self.n_example):
                        qa_pairs.append((env.question, env.solution))
                questions_new = self.multiple_processes_gpt(qa_pairs, generate_new_math_question)
               
                self.print_log('------GPT starts to generate new answer--------')
                new_envs = []
                # to make sure the size of buffers the same for each rank
                valid_questions_new = [q if q != 'error' else random.sample(self.envs, 1)[0].question for q in questions_new ]
                gt_qa_pairs_new = self.multiple_processes_gpt(valid_questions_new, generate_math_answer)
                for gt_qa_pair_new in gt_qa_pairs_new:
                    if gt_qa_pair_new[1] != 'error':
                        new_response, new_answer = extract_answer(gt_qa_pair_new[1]+'\n')
                        
                        new_env = GSM8kEnv(question=gt_qa_pair_new[0], 
                                            key=new_answer,
                                            tokenizer=self.tokenizer)
                        new_env.gt_solution = new_response
                        new_envs.append(new_env)
                    else:
                        # to make sure the size of buffers the same for each rank
                        env = random.sample(self.envs, 1)[0]
                        new_env = GSM8kEnv(question=env.question, 
                                            key=env.answer,
                                            tokenizer=self.tokenizer)
                        new_env.gt_solution = env.gt_solution
                        new_envs.append(new_env)
                self.print_log('------new generated ' + str(len(new_envs)) + ' envs --------')
                self.run(new_envs)
                ppo_buffers_1 = [env.buffer for env in new_envs]
                if self.use_gt_solution:
                    
                    ppo_buffers_2 = [env.get_gt_buffer() for env in self.envs]
                    
                    ppo_buffers_3 = [env.get_gt_buffer() for env in new_envs]
                    
            # assert len(ppo_buffers_0) == len(ppo_buffers_1) == len(ppo_buffers_2) == len(ppo_buffers_3)
            for i in range(len(ppo_buffers_0)):
                self.ppo_buffers.append(ppo_buffers_0[i])
                if ppo_buffers_1:
                    self.ppo_buffers.append(ppo_buffers_1[i])
                if ppo_buffers_2:
                    self.ppo_buffers.append(ppo_buffers_2[i])
                if ppo_buffers_3:
                    self.ppo_buffers.append(ppo_buffers_3[i])

        answers = [env.answer for env in self.envs]
        correct = [env.is_correct() for env in self.envs]
        log += f'Finished evaluating, Correct: {sum(correct)}, out of {len(self.envs)}, accuracy: {sum(correct)/len(self.envs)}'
        self.print_log(log)
        if test:
            log_path = os.path.join(self.OUTPUT_DIR, 'log', 'test')
        else:
            log_path = os.path.join(self.OUTPUT_DIR, 'log', 'train')
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok = True)
        with open(os.path.join(log_path, f'{len(self.envs)}_questions_{n_iter}_iter_rank{self.global_rank}.txt'), 'w') as f:
            f.write(log)
        self.llm.cpu()
        del self.llm # free memory
        return answers
    
    def print_log(self, log_str):
        with open(self.LOG_PATH, 'a') as wf:
            wf.write(log_str + '\n')

    def prompt_llm(self, queries):
        llm_outputs = []
        for q in tqdm(queries):
            llm_outputs.append(self.predict(q, 0.0, 1, self.max_tokens))
        return llm_outputs


    def predict(self,
                actions,
                top_p,
                temperature,
                max_length_tokens):
        
        fake_input_ids = self.tokenizer('fake example', return_tensors="pt")["input_ids"].to(self.llm.device)
        # input_ids = self.tokenizer.encode(new_text)
        input_ids = torch.tensor([actions]).type_as(fake_input_ids)

        # "\n#### 1234\n"
        stop_words = ["####", "\n"] # A and B instead of A or B in gsm8k
        with torch.no_grad():
            text, tokens, logprobs, state_values =self.greedy_search(input_ids,stop_words=stop_words,max_length=max_length_tokens,temperature=temperature,top_p=top_p)
            # for x in self.greedy_search(input_ids,stop_words=stop_words,max_length=max_length_tokens,temperature=temperature,top_p=top_p):
            #     if is_stop_word_or_prefix(x,stop_words) is False:
                    # x = x.strip(" ")
        return text, tokens, logprobs, state_values
        
    # Greedy Search
    def greedy_search(
            self,
            input_ids: torch.Tensor,
            stop_words: list,
            max_length: int,
            temperature: float = 1.0,
            top_p: float = 1.0):
        generated_tokens = []
        generated_tokens_logprobs = []
        generated_state_values = []
        past_key_values = None
        # current_length = 1
        for i in range(max_length):
            with torch.no_grad():
                if past_key_values is None:
                    outputs = self.llm(input_ids)
                else:
                    outputs = self.llm(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits[:, -1, :]
                state_values = outputs.state_values[:, -1]
                past_key_values = outputs.past_key_values

            # apply temperature
            logits /= temperature

            probs = torch.softmax(logits, dim=-1)
            # apply top_p 
            # TODO top_p is ok for PPO?
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
            input_ids = torch.cat((input_ids, next_token), dim=-1)

            generated_tokens.append(next_token[0].item())
            # print(probs, next_token)
            generated_tokens_logprobs.append(probs[:, next_token[0].item()].log().item())
            generated_state_values.append(state_values.item()) # TODO append next one or current one?
            text = self.tokenizer.decode(generated_tokens)

            # # yield text
            # if any([x in text for x in stop_words]):
            if stop_words[0] in text and text.endswith(stop_words[1]):
                return text, generated_tokens, generated_tokens_logprobs, generated_state_values
            
        return text, generated_tokens, generated_tokens_logprobs, generated_state_values

    


class GSM8kEnv:
    def __init__(self,
                 question: str,
                 key: str,
                 tokenizer,
                 max_steps: int = 1,
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.solution = ''
        self.gt_solution = ''
        self.key = key
        self.max_steps = max_steps
        self.CUTOFF_LEN = 1024 #TODO increase?
        self.tokenizer = tokenizer
        self.__reset_agent()

    
    def update_scratchpad(self, text, tokens=None, logprobs=None, state_values=None, mask=1) -> None:
        self.scratchpad += text
        actions = self.tokenizer.encode(text)[1:]

        if tokens is None:
            for i, a in enumerate(actions):
                action = a
                logprob = 1.
                reward = 0.
                state_value = 0. 
                system_mask = mask
                self.buffer.append(action, logprob, reward, state_value, system_mask)
        else:
            assert len(tokens) == len(logprobs) == len(state_values)
            if len(tokens) < len(actions) - 1:
                print('xxxxxx'*10)
                print('text: ', text, 'actions:', actions, 'tokens:', tokens)
                print('xxxxxx'*10)
            # for i, a in enumerate(actions):
            temp_tokens = []
            for i, a in enumerate(tokens):
                action = a
                logprob = logprobs[i]
                reward = 0.
                state_value = state_values[i]
                system_mask = mask
                self.buffer.append(action, logprob, reward, state_value, system_mask)
                temp_tokens.append(a)
                if text and len(tokens) > len(actions) + 1:
                    valid_actions = self.tokenizer.decode(temp_tokens)
                    if text in valid_actions or i == len(tokens) - 1:
                        # print('++++'*10)
                        # print('encode_actions: ', text)
                        # print('append_actions: ', valid_actions)
                        # print('generated_text: ', self.tokenizer.decode(tokens))
                        # print('++++'*10)
                        break

    
    def _init_scratchpad(self) -> str:
        return create_prompt(self.question)
    
    def query_answer(self):
        # self.update_scratchpad("\n### Answer:")
        self.update_scratchpad("\n####")
        return self.buffer.actions
    
    def query_assistant(self):
        self.update_scratchpad("\n### Assistant:")
        return self.buffer.actions
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return self.answer == self.key

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.buffer.actions) > self.CUTOFF_LEN)) and not self.finished
     
    def get_gt_buffer(self):
        temp_buffer = self.buffer
        temp_scratchpad = self.scratchpad
        self.buffer = RolloutBuffer()
        self.scratchpad = ''
        self.update_scratchpad(self._init_scratchpad())
        self.query_assistant()
        self.update_scratchpad(self.gt_solution, mask=2)
        gt_buffer = self.buffer
        if self.gt_solution == 'error':
            gt_buffer.rewards[-1] = -1.
        else:
            gt_buffer.rewards[-1] = 1.
        self.buffer = temp_buffer
        self.scratchpad = temp_scratchpad
        return gt_buffer

    def __reset_agent(self) -> None:
        # self.max_tokens = 512
        self.max_tokens = 512
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''
        self.buffer = RolloutBuffer()
        self.update_scratchpad(self._init_scratchpad())
            
    def _build_agent_prompt(self) -> str:
        return self.scratchpad



