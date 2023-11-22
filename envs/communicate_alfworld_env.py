import re, string, os, sys
from typing import List, Union, Literal, Any, Dict, Tuple
from enum import Enum
import os.path


import copy
import json
import yaml
import importlib
import alfworld
import alfworld.agents.environment
from env_history import EnvironmentHistory

import os
import requests
import ast
import time
import json
import random
import pickle
import torch
import threading
from tqdm import tqdm


import torch.distributed as dist

from iterative_ppo import PPOTrainer, PPODataset, PPODataCollatorForLanguageModeling, ActorCriticLlama, RolloutBuffer

FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


class ReactAlfworldCoordinator:
    def __init__(self,
                 id_list,
                 llm,
                 tokenizer,
                 output_dir,
                 ) -> None:
        
        self.max_tokens = 128
        self.n_example = 0
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.global_rank = dist.get_rank() if world_size != 1 else 0
        self.OUTPUT_DIR = output_dir
        self.ppo_buffers = []
        
        
        self.init_log_path = os.path.join('./llama_exp/root/alfworld', 'txt003_trial_0.txt')
        self.envs = []
        
        # initialize environment configs
        self.env_configs: List[Dict[str, Any]] = []
        self.id_list = id_list
        for i in id_list:
            self.env_configs += [{
                'name': f'env_{i}',
                'memory': [],
                'is_success': False,
                'skip': False
            }]
    

        #self.set_llm(llm=llm, tokenizer=tokenizer)
        self.set_llm(llm=copy.deepcopy(llm), tokenizer=tokenizer)
        # self.success_examples = self.log2examples(self.init_log_path)
        
    
    def set_llm(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer

        
    def alfworld_run(self, env, base_prompt, memory: List[str], to_print=True, ob='') -> Tuple[EnvironmentHistory, bool]:
        
        if len(memory) > 3:
            env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
        else:
            env_history = EnvironmentHistory(base_prompt, ob, memory, [])
        env_history.reset()
        # init_prompt = prompt + ob + '\n>'
        # prompt = ''
        env_buffer = ReactAlfworldEnv(prompt=str(env_history) + ">", tokenizer=self.tokenizer)
        if to_print:
            print(ob)
            sys.stdout.flush()
        cur_step = 0
        #while cur_step < 50:
        while cur_step < 50 and not env_buffer.is_halted() and not env_buffer.is_finished():
            # action = llm(init_prompt + prompt, stop=['\n']).strip()
            text, tokens, logprobs, state_values = self.predict(env_buffer.buffer.actions, 0.0, 1, self.max_tokens)
            action = text.strip()
            env_history.add("action", action)
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            if action.startswith('think:') or action.startswith('hint:'):
                observation = 'OK.'
                # observation = ''
                # if to_print:
                #     print(f'>{action}')
                #     sys.stdout.flush()
            # else:
            #     env_history.add("observation", observation)
                # if to_print:
                #     print(f'> {action}\n{observation}')
                #     sys.stdout.flush()
            env_history.add("observation", observation)
            env_buffer.update_scratchpad(action, tokens, logprobs, state_values, mask=0) 
            
            
            if to_print:
                print(f'> {action}\n{observation}')
                sys.stdout.flush()
            # prompt += f' {action}\n{observation}\n>'
            if done:
                #print(str(env_history) + "> reward:", reward)
                if reward:
                    env_buffer.buffer.rewards[-1] = 1.
                else:
                    env_buffer.buffer.rewards[-1] = -1.
                env_buffer.finished = True
                self.envs.append(env_buffer)
                return env_history, reward
            elif env_history.check_is_exhausted():
                env_buffer.buffer.rewards[-1] = -1.
                env_buffer.finished = True
                self.envs.append(env_buffer)
                return env_history, False
            elif env_buffer.is_halted():
                env_buffer.buffer.rewards[-1] = -1.
                self.envs.append(env_buffer)
                return env_history, False
            env_buffer.update_scratchpad('\n'+ observation + '\n> ')
            env_buffer.step_n += 1
            cur_step += 1
        env_buffer.buffer.rewards[-1] = -1.
        self.envs.append(env_buffer)
        return env_history, False

            
    
    def ppo_trial_generate(self, n_iter, test=True):
        importlib.reload(alfworld)
        importlib.reload(alfworld.agents.environment)
        self.llm.eval()

        with open('llama_exp/base_config.yaml') as reader:
            config = yaml.safe_load(reader)
        if test:
            split = "eval_out_of_distribution"
            self.trial_log_path = os.path.join(self.OUTPUT_DIR, 'log', f'test_trial_iter{n_iter}_rank{self.global_rank}.log')
            self.world_log_path = os.path.join(self.OUTPUT_DIR, 'log', f'test_world_iter{n_iter}_rank{self.global_rank}.log')
        else:
            split = "train"
            self.trial_log_path = os.path.join(self.OUTPUT_DIR, 'log', f'train_trial_iter{n_iter}_rank{self.global_rank}.log')
            self.world_log_path = os.path.join(self.OUTPUT_DIR, 'log', f'train_world_iter{n_iter}_rank{self.global_rank}.log')
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'log')):
            os.makedirs(os.path.join(self.OUTPUT_DIR, 'log'), exist_ok = True)
        
        env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=1)

        num_successes: int = 0
        num_additional_successes: int = 0
        num_envs: int = len(self.env_configs)
        

        if test:
            num_total_envs = 134
            # env.seed(self.global_rank)

        else:
            # num_total_envs = 3553
            #num_total_envs = 256
            #num_total_envs = 1024
            # env.seed(sum(self.id_list))
            num_total_envs = len(self.id_list)
            env.seed(self.id_list[-1])
            self.id_list = list(range(num_total_envs))

        # for z, env_config in enumerate(self.env_configs):
        for z in range(num_total_envs):
            ob, info = env.reset()
            ob = '\n'.join(ob[0].split('\n\n')[1:])
            name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
            print(f"using {name}")

            if z in self.id_list:
               idx = self.id_list.index(z)
               env_config = self.env_configs[idx]
            else:
               continue
            if env_config["is_success"]:
                num_successes += 1
            
                # log to world log
                with open(self.world_log_path, 'a') as wf:
                    wf.write(f'Environment #{z} Trial #0: SUCCESS\n')
                with open(self.trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
                continue
            
            for i, (k, v) in enumerate(PREFIXES.items()):
                if name.startswith(k):
                    # base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
                    base_prompt = 'Interact with a household to solve a task.'
                    final_env_history, is_success = self.alfworld_run(env, base_prompt, [], to_print=True, ob=ob)

                    # update env config
                    if is_success:
                        status_str: str = f'Environment #{z} Trial #0: SUCCESS'
                        # self.env_configs[z]['is_success'] = True
                        self.env_configs[idx]['is_success'] = True
                        num_successes += 1
                        num_additional_successes += 1
                    else:
                        status_str: str = f'Environment #{z} Trial #0: FAIL'

                    # log to world log
                    with open(self.world_log_path, 'a') as f:
                        f.write(status_str + '\n')

                    # log env results to trial log
                    with open(self.trial_log_path, 'a') as wf:
                        wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

        
        if not test:
            ppo_buffers_0 = [env.buffer for env in self.envs]
            ppo_buffers_1 = []
            for i in range(len(ppo_buffers_0)):
                self.ppo_buffers.append(ppo_buffers_0[i])

        # close environment object
        env.close()

        # log trial results to trial and world logs
        log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
        with open(self.trial_log_path, 'a') as wf:
            wf.write(log_str)
        with open(self.world_log_path, 'a') as wf:
            wf.write(log_str + '\n')
        
        print(log_str)

        self.llm.cpu()
        del self.llm # free memory
        



    def predict(self,
                actions,
                top_p,
                temperature,
                max_length_tokens):
        
        fake_input_ids = self.tokenizer('fake example', return_tensors="pt")["input_ids"].to(self.llm.device)
        # input_ids = self.tokenizer.encode(new_text)
        input_ids = torch.tensor([actions]).type_as(fake_input_ids)

    
        # stop_words = ["[|Human|]", "[|AI|]"]
        stop_words = ["\n", "\n\n"]
        # stop_words = ["\n\n\n", "\n\n"]
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
                    # outputs = self.llm(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                    outputs = self.llm(input_ids[:, -1:], past_key_values=past_key_values)
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



    def get_instruction_finetune_data(self, init_n_train=1024):
       
        
        success_examples = [d[f'react_{v}_1'] for i, (k, v) in enumerate(PREFIXES.items())]
        success_examples += [d[f'react_{v}_0'] for i, (k, v) in enumerate(PREFIXES.items())]
        fail_examples = []
        
        with open(self.init_log_path, 'r') as f:
            
            example = 'Interact with a household to solve a task.\nHere is the task:\n'
            last_line = ''
            line_idx = 0
            for line in f:
                # if self.global_rank == 0:
                #     print(f'line{line_idx}: {line}')
                line_idx += 1
                if 'think:' in line and 'more likely' in line:
                    # cur_line = line.replace('think:', 'hint:')
                    cur_line = line
                else:
                    cur_line = line

                # if '> ' in cur_line:
                #     cur_line = cur_line.replace('> ', '>>> ')
                
                
                if 'please retry' in line or 'please retry' in last_line:
                    last_line = line
                    continue
                # if 'OK.' in line:
                #     # example += '\n'
                #     continue

                if line.startswith('Here is the task:'):
                    example = 'Interact with a household to solve a task.\nHere is the task:\n'
                
                elif line.startswith('STATUS:'):
                    if 'FAIL' in line or not last_line.startswith('You'):
                    # if 'FAIL' in line:
                        # if not last_line.startswith('You'):
                        #     flag = last_line.startswith('You')
                        #     print(f'last_line {flag}: {last_line} ')
                        fail_examples.append(example + '\n')
                    else:
                        # print(example + '\n')
                        success_examples.append(example + '\n')
                    example = ''
                else:
                    example += cur_line
                
                if len(line) > 1:
                    last_line = line
                # else:
                    # print(f'line{line_idx}: {line}')
                
                
        
        print('len of success_examples: ', len(success_examples))
        if len(success_examples) < init_n_train:
            return random.choices(success_examples, k=init_n_train)
        return random.sample(success_examples, init_n_train)
        
                    


class ReactAlfworldEnv:
    def __init__(self,
                 prompt: str,
                 tokenizer,
                 max_steps: int = 32,
                 ) -> None:
        
        self.prompt = prompt
        self.max_steps = max_steps
        self.CUTOFF_LEN = 2048 #TODO increase?
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
                        print('++++'*10)
                        print('encode_actions: ', text)
                        print('append_actions: ', valid_actions)
                        print('generated_text: ', self.tokenizer.decode(tokens))
                        print('++++'*10)
                        break

    def print_scratchpad(self):
        if self.scratchpad.endswith('\n'):
            print(self.scratchpad.split('\n')[-2])
        else:
            print(self.scratchpad.split('\n')[-1])
    
    def query_think(self):
        self.update_scratchpad(f'\nThought {self.step_n}:')
        return self.buffer.actions
    
    def query_act(self):
        self.update_scratchpad(f'\nAction {self.step_n}:')
        return self.buffer.actions
    
    def _init_scratchpad(self) -> str:
        return self.prompt

    def is_finished(self) -> bool:
        return self.finished


    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.buffer.actions) > self.CUTOFF_LEN)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''
        self.buffer = RolloutBuffer()
        self.update_scratchpad(self._init_scratchpad())
            
