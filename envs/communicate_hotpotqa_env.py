import re, string, os, sys
from typing import List, Union, Literal
from enum import Enum
import os.path

import tiktoken
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore.base import Docstore
from langchain.prompts import PromptTemplate
from prompts import react_agent_prompt
from prompts import react_fewshots_prompt
from fewshots import WEBTHINK_SIMPLE6, WEBTHINK_SIMPLE16, WEBTHINK_SIMPLE2



import copy
import os
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


from util import summarize_react_trial, log_react_trial, save_agents
import torch.distributed as dist

from iterative_ppo import PPOTrainer, PPODataset, PPODataCollatorForLanguageModeling, ActorCriticLlama, RolloutBuffer



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


def generate_gpt_response(inputs):
    if type(inputs) == str:
        query = inputs
        student_response=None
        question = query.split('Question: ')[1].split('\n')[0]
        content = ""
    else:
        question, query, student_response = inputs
        content = f"Here is an problem and the solution of the student: \n{student_response}\n Please generate a a better solution to solve the question. \n Please start directly:\n "
    system_message = '''Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.'''
    
    # response, history = api(content + query, system_message)
    response = gen_from_gpt(content + query, system_message)
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



class ReactQACoordinator:
    def __init__(self,
                 json_data,
                 llm,
                 tokenizer,
                 output_dir,
                 docstore: Docstore = Wikipedia(),
                 ) -> None:
        
        self.max_tokens = 256
        # self.max_tokens = 128
        self.n_example = 0
        self.global_rank = dist.get_rank()
        self.OUTPUT_DIR = output_dir
        self.ppo_buffers = []
        self.gpt_feedback = False
        self.self_feedback = False
        self.deep_copy_llm = True
        self.model_device = llm.device
        self.init_log_path = os.path.join('./data/hotpotqa', '10000_questions_1_trials.txt')
        self.LOG_PATH = os.path.join(self.OUTPUT_DIR, 'log', 'print', f'print_log_rank{self.global_rank}.txt')
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'log', 'print')):
            os.makedirs(os.path.join(self.OUTPUT_DIR, 'log', 'print'), exist_ok = True)
        
        self.docstore = DocstoreExplorer(docstore) # Search, Lookup
        if self.deep_copy_llm:
            self.set_llm(llm=copy.deepcopy(llm), tokenizer=tokenizer)
        else:
            self.set_llm(llm=llm, tokenizer=tokenizer)
        self.success_examples = self.log2examples(self.init_log_path)
        self.envs = [ReactQAEnv(question=row['question'], 
                                key=row['answer'],
                                tokenizer=tokenizer,
                                react_examples=self.insert_examples(self.success_examples, self.n_example, simple=True)) for row in json_data] 
        if self.gpt_feedback:
            self.gpt_envs = [ReactQAEnv(question=row['question'], 
                                key=row['answer'],
                                tokenizer=tokenizer,
                                react_examples=self.insert_examples(self.success_examples, self.n_example, simple=True)) for row in json_data] 
            
        if self.self_feedback:
            self.self_envs = [ReactQAEnv(question=row['question'], 
                                key=row['answer'],
                                tokenizer=tokenizer,
                                react_examples=self.insert_examples(self.success_examples, self.n_example, simple=True)) for row in json_data] 
    
    
    
    
    def set_llm(self, llm, tokenizer):
        self.llm = llm.cpu()
        # self.llm = llm
        self.tokenizer = tokenizer

    def run(self, envs):
        
        active_envs = [env for env in envs if not (env.is_finished() or env.is_halted())]
        
        while active_envs:
            # Think
            queries_think = []
            for i, env in enumerate(active_envs):
                queries_think.append(env.query_think()) 
            outputs_think = self.prompt_llm(queries_think) # TODO parallel with vllm
            for i, env in enumerate(active_envs):
                env.update_scratchpad(*outputs_think[i], mask=0)
                env.print_scratchpad(self.LOG_PATH)

            # Act
            queries_act = []
            for i, env in enumerate(active_envs):
                queries_act.append(env.query_act())
            outputs_act = self.prompt_llm(queries_act) # TODO parallel with vllm
            
            for i, env in enumerate(active_envs):
                text, tokens, logprobs, state_values = outputs_act[i]
                action_str = text
                action = action_str[:action_str.find(']')+1] # TODO debug action parsing
                env.update_scratchpad(action, tokens, logprobs, state_values, mask=0) 
                parsed_action = self.parse_action(action)
                if parsed_action:
                    action_type, argument = parsed_action
                else:
                    self.print_log('----'*10)
                    self.print_log('action_str: ' + action_str)
                    self.print_log('----'*10)
                    action_type, argument = 'Finish', 'N/A'
                
                env.print_scratchpad(self.LOG_PATH)
                
                # Observe
                env.update_scratchpad(f'\nObservation {env.step_n}:')
                if action_type == 'Finish':
                    env.answer = argument
                    if env.is_correct():
                        env.update_scratchpad('Answer is CORRECT')
                        env.buffer.rewards[-1] = 1.
                        # print('Answer is CORRECT')
                    else: 
                        env.update_scratchpad('Answer is INCORRECT')
                        env.buffer.rewards[-1] = -1.
                        # print('Answer is INCORRECT')
                    env.finished = True
                    # print(f'Answer: {env.key}')
                    with open(self.LOG_PATH, 'a') as wf:
                        wf.write(f'Answer: {env.key}\n')
                elif action_type == 'Search':
                    try:
                        search_output = self.format_step(self.docstore.search(argument), max_len=self.max_tokens)
                        env.update_scratchpad(search_output)
                    except Exception as e:
                        self.print_log('----'*10)
                        self.print_log('search error: ')
                        self.print_log('search argument: ' + argument)
                        self.print_log('----'*10)
                        print(e)
                        env.update_scratchpad(f'Could not find that page, please try again.')
                
                elif action_type == 'Lookup':
                    try:
                        lookup_output = self.format_step(self.docstore.lookup(argument), max_len=self.max_tokens)
                        env.update_scratchpad(lookup_output)
                    except ValueError:
                        env.update_scratchpad(f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.')

                else:
                    env.update_scratchpad('Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].')
                
                env.print_scratchpad(self.LOG_PATH)
                env.step_n += 1
                if env.is_halted() and not env.is_finished():
                    env.buffer.rewards[-1] = -1.

            active_envs = [env for env in envs if not (env.is_finished() or env.is_halted())]
            

    def run_gpt(self, envs):
        active_envs = [env for env in envs if not (env.is_finished() or env.is_halted())]
        while active_envs:
            # Think
            queries_think = []
            for i, env in enumerate(active_envs):
                question = env.question
                query = env.query_think(return_str=True)
                student_response = env.solution
                queries_think.append((question, query, student_response)) 
            outputs_think = self.multiple_processes_gpt(queries_think, generate_gpt_response) # TODO parallel with vllm
            outputs_think_dict = {output_think[0]: output_think[1] for output_think in outputs_think}
            for i, env in enumerate(active_envs):
                think = outputs_think_dict[env.question].split('\n')[0]
                env.update_scratchpad(think, mask=2)
                env.print_scratchpad(self.LOG_PATH)

            # Act
            queries_act = []
            for i, env in enumerate(active_envs):
                question = env.question
                query = env.query_act(return_str=True)
                student_response = env.solution
                queries_act.append((question, query, student_response))
            outputs_act = self.multiple_processes_gpt(queries_act, generate_gpt_response) # TODO parallel with vllm
            outputs_act_dict = {output_act[0]: output_act[1] for output_act in outputs_act}
            
            for i, env in enumerate(active_envs):
                action_str = outputs_act_dict[env.question]
                action = action_str[:action_str.find(']')+1] # TODO debug action parsing
                env.update_scratchpad(action, mask=2) 
                parsed_action = self.parse_action(action)
                if parsed_action:
                    action_type, argument = parsed_action
                else:
                    self.print_log('----'*10)
                    self.print_log('action_str: ' + action_str)
                    self.print_log('----'*10)
                    action_type, argument = 'Finish', 'N/A'
                
                env.print_scratchpad(self.LOG_PATH)
                
                # Observe
                env.update_scratchpad(f'\nObservation {env.step_n}:')
                if action_type == 'Finish':
                    env.answer = argument
                    if env.is_correct():
                        env.update_scratchpad('Answer is CORRECT')
                        env.buffer.rewards[-1] = 1.
                        # print('Answer is CORRECT')
                    else: 
                        env.update_scratchpad('Answer is INCORRECT')
                        env.buffer.rewards[-1] = -1.
                        # print('Answer is INCORRECT')
                    env.finished = True
                    # print(f'Answer: {env.key}')
                    with open(self.LOG_PATH, 'a') as wf:
                        wf.write(f'Answer: {env.key}\n')
                elif action_type == 'Search':
                    try:
                        search_output = self.format_step(self.docstore.search(argument), max_len=self.max_tokens)
                        env.update_scratchpad(search_output)
                    except Exception as e:
                        self.print_log('----'*10)
                        self.print_log('search error: ')
                        self.print_log('search argument: ' + argument)
                        self.print_log('----'*10)
                        print(e)
                        env.update_scratchpad(f'Could not find that page, please try again.')
                
                elif action_type == 'Lookup':
                    try:
                        lookup_output = self.format_step(self.docstore.lookup(argument), max_len=self.max_tokens)
                        env.update_scratchpad(lookup_output)
                    except ValueError:
                        env.update_scratchpad(f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.')

                else:
                    env.update_scratchpad('Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].')
                
                env.print_scratchpad(self.LOG_PATH)
                env.step_n += 1
                if env.is_halted() and not env.is_finished():
                    env.buffer.rewards[-1] = -1.

            active_envs = [env for env in envs if not (env.is_finished() or env.is_halted())]
            

    def run_gpt_communication(self, envs):
        active_envs = [env for env in envs if not (env.is_finished() or env.is_halted())]
        while active_envs:
            # Think
            queries_think = []
            for i, env in enumerate(active_envs):
                question = env.question
                query = env.query_think(return_str=True)
                student_response = env.solution
                queries_think.append((question, query, student_response)) 
            outputs_think = self.multiple_processes_gpt(queries_think, generate_gpt_response) # TODO parallel with vllm
            outputs_think_dict = {output_think[0]: output_think[1] for output_think in outputs_think}
            for i, env in enumerate(active_envs):
                think = outputs_think_dict[env.question].split('\n')[0]
                env.update_scratchpad(think, mask=2)
                env.print_scratchpad(self.LOG_PATH)

            # Act
            queries_act = []
            for i, env in enumerate(active_envs):
                queries_act.append(env.query_act())
            outputs_act = self.prompt_llm(queries_act) # TODO parallel with vllm
            
            for i, env in enumerate(active_envs):
                text, tokens, logprobs, state_values = outputs_act[i]
                action_str = text
                action = action_str[:action_str.find(']')+1] # TODO debug action parsing
                env.update_scratchpad(action, tokens, logprobs, state_values, mask=0) 
                parsed_action = self.parse_action(action)
                if parsed_action:
                    action_type, argument = parsed_action
                else:
                    self.print_log('----'*10)
                    self.print_log('action_str: ' + action_str)
                    self.print_log('----'*10)
                    action_type, argument = 'Finish', 'N/A'
                
                env.print_scratchpad(self.LOG_PATH)
                
                # Observe
                env.update_scratchpad(f'\nObservation {env.step_n}:')
                if action_type == 'Finish':
                    env.answer = argument
                    if env.is_correct():
                        env.update_scratchpad('Answer is CORRECT')
                        env.buffer.rewards[-1] = 1.
                        # print('Answer is CORRECT')
                    else: 
                        env.update_scratchpad('Answer is INCORRECT')
                        env.buffer.rewards[-1] = -1.
                        # print('Answer is INCORRECT')
                    env.finished = True
                    # print(f'Answer: {env.key}')
                    with open(self.LOG_PATH, 'a') as wf:
                        wf.write(f'Answer: {env.key}\n')
                elif action_type == 'Search':
                    try:
                        search_output = self.format_step(self.docstore.search(argument), max_len=self.max_tokens)
                        env.update_scratchpad(search_output)
                    except Exception as e:
                        self.print_log('----'*10)
                        self.print_log('search error: ')
                        self.print_log('search argument: ' + argument)
                        self.print_log('----'*10)
                        print(e)
                        env.update_scratchpad(f'Could not find that page, please try again.')
                
                elif action_type == 'Lookup':
                    try:
                        lookup_output = self.format_step(self.docstore.lookup(argument), max_len=self.max_tokens)
                        env.update_scratchpad(lookup_output)
                    except ValueError:
                        env.update_scratchpad(f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.')

                else:
                    env.update_scratchpad('Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].')
                
                env.print_scratchpad(self.LOG_PATH)
                env.step_n += 1
                if env.is_halted() and not env.is_finished():
                    env.buffer.rewards[-1] = -1.

            active_envs = [env for env in envs if not (env.is_finished() or env.is_halted())]

    def multiple_processes_gpt(self, tasks, func):
        # TODO check the order of outputs
        outputs = []
        # batch_size = 16
        batch_size = 1
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
        self.llm.to(self.model_device)
        self.llm.eval()
        log = ''
        self.run(self.envs)
        log += log_react_trial(self.envs, 1)
        correct, incorrect, halted = summarize_react_trial(self.envs)
        self.print_log(f'Finished LLaMA generation, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
                    
        if not test:
            ppo_buffers_0 = [env.buffer for env in self.envs]
            ppo_buffers_1 = []
            ppo_buffers_2 = []
            if self.gpt_feedback:
                self.print_log('------GPT starts to do correction--------')
                for i in range(len(self.envs)):
                    self.gpt_envs[i].solution = self.envs[i].get_solution()
                self.run_gpt_communication(self.gpt_envs)
                ppo_buffers_1 = [env.buffer for env in self.gpt_envs]
                correct, incorrect, halted = summarize_react_trial(self.gpt_envs)
                self.print_log(f'Finished GPT generation, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
            if self.self_feedback:
                self.print_log('------LLM starts to self-feedback--------')
                for i in range(len(self.envs)):
                    cur_solution = self.envs[i].get_solution()
                    reflexion_prompt = '\nThought 1:' + cur_solution.split('\nThought 1:')[-1]
                    reflexion_prompt += '\nPlease improve the above process to get a better solution to solve the question.'
                    reflexion_prompt += '\nQuestion: ' + self.envs[i].question 
                    self.self_envs[i].update_scratchpad(reflexion_prompt, mask=-1) # mask it out for training
                self.run(self.self_envs)
                ppo_buffers_1 = [env.buffer for env in self.self_envs]
                correct, incorrect, halted = summarize_react_trial(self.self_envs)
                self.print_log(f'Finished Self-feedback generation, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
        
            for i in range(len(ppo_buffers_0)):
                self.ppo_buffers.append(ppo_buffers_0[i])
                if ppo_buffers_1:
                    self.ppo_buffers.append(ppo_buffers_1[i])
                if ppo_buffers_2:
                    self.ppo_buffers.append(ppo_buffers_1[i])
            
  
        if test:
            log_path = os.path.join(self.OUTPUT_DIR, 'log', 'test')
        else:
            log_path = os.path.join(self.OUTPUT_DIR, 'log', 'train')
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok = True)
        with open(os.path.join(log_path, f'{len(self.envs)}_questions_{n_iter}_iter_rank{self.global_rank}.txt'), 'w') as f:
            f.write(log)
        if self.deep_copy_llm:
            self.llm.cpu()
            del self.llm # free memory
            # with torch.no_grad():
            #     torch.cuda.empty_cache()
        
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
        
        fake_input_ids = self.tokenizer('fake example', return_tensors="pt")["input_ids"].to(self.model_device)
        # input_ids = self.tokenizer.encode(new_text)
        input_ids = torch.tensor([actions]).type_as(fake_input_ids)

    
        # stop_words = ["[|Human|]", "[|AI|]"]
        stop_words = ["\n", "\n\n"]
        # stop_words = ["\n\n\n", "\n\n"]
        with torch.no_grad():
            text, tokens, logprobs, state_values =self.greedy_search(input_ids,stop_words=stop_words,max_length=max_length_tokens,temperature=temperature,top_p=top_p)
           
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

            # yield text
            if any([x in text for x in stop_words]):
                return text, generated_tokens, generated_tokens_logprobs, generated_state_values
            
        return text, generated_tokens, generated_tokens_logprobs, generated_state_values

    
    def parse_action(self, string):
        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string.strip())
        
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        
        else:
            return None


        
    def format_step(self, step: str, max_len=-1) -> str:
        if max_len > 0:
            temp_toks = self.tokenizer.encode(step)
            if len(temp_toks) > max_len:
            # if len(step) > max_len:
                step = self.tokenizer.decode(temp_toks[1:max_len+1])
        return step.strip('\n').strip().replace('\n', '')
    

    def insert_examples(self, candidates, n=2, simple=False):
        if n < 1:
            return ''
        assert len(candidates) >= n
        if simple:
            fewshots_prompt = ''
        else:
            fewshots_prompt = 'Here are some examples:\n'

        for e in random.sample(candidates, n):
            fewshots_prompt += e
        
        if simple:
            pass
        else:
            # fewshots_prompt += '\n(END OF EXAMPLES)\n\n\n'
            fewshots_prompt += '(END OF EXAMPLES)\n'
        return fewshots_prompt
    

    def get_instruction_finetune_data(self, init_n_train=1024):
        if self.success_examples is None:
            self.success_examples = ['Question:'+ e.strip('\n') + '\n' + '\n'  for e in WEBTHINK_SIMPLE16.split('Question:')[1:]]
        

        self.success_chat_data = []
        train_chat_data = []
        for i in range(init_n_train):

            react_examples = ""
            for e in random.sample(self.success_examples, self.n_example):
                react_examples += e
            sample = ""
            for s in random.sample(self.success_examples, 1):
                sample += s
            fewshots_sample = react_fewshots_prompt.format(examples = react_examples,
                                         sample = sample)
            train_chat_data.append(fewshots_sample)
        return train_chat_data
        
        

    def log2examples(self, log_path):
        success_examples = ['Question:'+ e.strip('\n') +'\n' + '\n' for e in WEBTHINK_SIMPLE6.split('Question:')[1:]]
        correct_examples_flag = True
        with open(log_path, 'r') as f:
            # text = ''
            example = ''
            for line in f:
                if line.startswith('------------- BEGIN INCORRECT'):
                    correct_examples_flag = False 
                if line.startswith('----------') or line.startswith('#########') or line.startswith('\n'):
                    continue
                if line.startswith('Trial summary:') or line.startswith('BEGIN TRIAL'):
                    continue
                if line.startswith('Correct answer:'):
                    if correct_examples_flag:
                        # success_examples.append(example + '\n')
                        success_examples.append(example)
                    # text = ''
                    example = ''
                else:
                    # text += line
                    # example += line
                    temp_toks = self.tokenizer.encode(line)
                    if len(temp_toks) > 256:
                        step = self.tokenizer.decode(temp_toks[1:257])
                    # if len(temp_toks) > 128:
                    #     step = self.tokenizer.decode(temp_toks[1:129])
                        example += step.strip('\n').strip().replace('\n', '') + '\n' # TODO check it
                    else:
                        example += line
                    
                if line.startswith('You may take as many steps as necessary.'):
                    example = ''
                    # text += self.insert_examples(self.success_examples, self.n_example)
        # self.examples2chat(self.train_size, success_examples)
        return success_examples


    def log2buffers(self, log_path):
        self.success_examples = self.log2examples(log_path)
        correct_examples_flag = True
        text = ''
        buffers = []
        with open(log_path, 'r') as f:
            for line in f:
                if line.startswith('------------- BEGIN INCORRECT'):
                    correct_examples_flag = False 
                if line.startswith('------------- BEGIN HALTED'):
                    halted_examples_flag = True 
                if line.startswith('----------') or line.startswith('#########') or line.startswith('\n'):
                    continue
                if line.startswith('Trial summary:') or line.startswith('BEGIN TRIAL'):
                    continue
                if line.startswith('Correct answer:'):
                    temp_toks = self.tokenizer.encode(text)
                    buffer = RolloutBuffer()
                    buffer.add_system_sequence(temp_toks[1:])
                    if correct_examples_flag:
                        buffer.rewards[-1] = 1.
                    # elif halted_examples_flag:
                    #     pass
                    else:
                        buffer.rewards[-1] = -1.
                    buffers.append(buffer)
                    text = ''
                    
                else:
                    temp_toks = self.tokenizer.encode(line)
                    if len(temp_toks) > self.max_tokens:
                        step = self.tokenizer.decode(temp_toks[1:self.max_tokens+1])
                        new_line = step.strip('\n').strip().replace('\n', '') + '\n' # TODO check it
                    else:
                        new_line = line
             
                    text += new_line
                    
                if line.startswith('You may take as many steps as necessary.'):
                    text += self.insert_examples(self.success_examples, self.n_example)
        return buffers 
        # assert len(buffers) >= self.train_size//self.world_size
        # self.ppo_buffers = random.sample(buffers, self.train_size//self.world_size)
                    


class ReactQAEnv:
    def __init__(self,
                 question: str,
                 key: str,
                 tokenizer,
                 react_examples: str = None,
                 max_steps: int = 6,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.solution = ''
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        self.CUTOFF_LEN = 4096 #TODO increase?
        # self.react_examples = WEBTHINK_SIMPLE6
        self.tokenizer = tokenizer
        self.react_examples = WEBTHINK_SIMPLE2 if react_examples is None else react_examples
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


    def get_solution(self):
        # anchor_str = "You may take as many steps as necessary."
        anchor_str = "(END OF EXAMPLES)"
        return self.scratchpad.split(anchor_str)[1]
    
    def print_scratchpad(self, log_dir=None):
        if self.scratchpad.endswith('\n'):
            info = self.scratchpad.split('\n')[-2]
        else:
            info = self.scratchpad.split('\n')[-1]
        if log_dir is None:
            print(info)
        else:
            with open(log_dir, 'a') as wf:
                wf.write(info  + '\n')

    def query_think(self, return_str=False):
        self.update_scratchpad(f'\nThought {self.step_n}:')
        if return_str:
            # return self.buffer.actions, self.get_solution()
            return self.get_solution()
        else:
            return self.buffer.actions
    
    def query_act(self, return_str=False):
        self.update_scratchpad(f'\nAction {self.step_n}:')
        if return_str:
            # return self.buffer.actions, self.get_solution()
            return self.get_solution()
        else:
            return self.buffer.actions
    
    def _init_scratchpad(self) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
                            question = self.question,
                            scratchpad = '')

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return normalize_answer(self.answer) == normalize_answer(self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.buffer.actions) > self.CUTOFF_LEN)) and not self.finished

    def __reset_agent(self) -> None:
        # self.max_tokens = 512
        self.max_tokens = 256
        # self.max_tokens = 128
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''
        self.buffer = RolloutBuffer()
        self.update_scratchpad(self._init_scratchpad())
            
    def _build_agent_prompt(self) -> str:
        # return self.agent_prompt.format(
        #                     examples = self.react_examples,
        #                     question = self.question,
        #                     scratchpad = self.scratchpad)
        return self.scratchpad

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

