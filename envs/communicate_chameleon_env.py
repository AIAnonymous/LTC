import re, string, os, sys
from typing import List, Union, Dict, Literal
from enum import Enum
import os.path

import uuid


from chameleon_example import chameleon_example, HARD_DEFAULT_TOPIC_CODES


from transformers.pipelines.conversational import Conversation, ConversationalPipeline
    


from tenacity import RetryError
from datetime import datetime, timedelta

from chatarena.agent import Player
from chatarena.backends import Human
from chatarena.environments.chameleon import Chameleon
# from chatarena.backends import OpenAIChat, TransformersConversational
# from chatarena.arena import Arena
from chatarena.message import SYSTEM_NAME as SYSTEM
from chatarena.message import Message

from chatarena.environments import Environment, TimeStep, load_environment

import collections

import copy
import os

import time
import json
import random
import torch
from tqdm import tqdm


import torch.distributed as dist

from iterative_ppo import PPOTrainer, PPODataset, PPODataCollatorForLanguageModeling, ActorCriticLlama, RolloutBuffer


class TooManyInvalidActions(Exception):
    pass

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"



class ChameleonCoordinator:
    def __init__(self,
                 json_data,
                 llm,
                 tokenizer,
                 output_dir,
                 ) -> None:
        
        self.max_tokens = 64
        self.num_players = 3
        self.global_rank = dist.get_rank()
        self.OUTPUT_DIR = output_dir
        self.ppo_buffers = []
        self.list_num_players = [3, 4, 5, 6]
       

        self.deep_copy_llm = True
        self.model_device = llm.device

        self.init_log_path = os.path.join('root/', 'chameleon', 'merged_trials.txt')
        self.LOG_PATH = os.path.join(self.OUTPUT_DIR, 'log', 'print', f'print_log_rank{self.global_rank}.txt')
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'log', 'print')):
            os.makedirs(os.path.join(self.OUTPUT_DIR, 'log', 'print'), exist_ok = True)
        
        if self.deep_copy_llm:
            self.set_llm(llm=copy.deepcopy(llm), tokenizer=tokenizer)
        else:
            self.set_llm(llm=llm.cpu(), tokenizer=tokenizer)

        self.success_examples = self.log2examples(self.init_log_path)

        self.envs = []
        for num_players in self.list_num_players:
            self.envs += [ChameleonEnv(llm=self.llm,
                                        tokenizer=tokenizer,
                                        num_players = num_players,
                                        log_path=self.LOG_PATH,
                                        seed=seed) for seed in json_data]
    
    def set_llm(self, llm, tokenizer):
        self.llm = llm.cpu()
        self.tokenizer = tokenizer


    def ppo_trial_generate(self, n_iter, test=True):
        self.llm.to(self.model_device)
        self.llm.eval()

        for env in tqdm(self.envs):
            env.run()
    
        if not test:
            for env in self.envs:
                self.ppo_buffers += env.buffers
            
        if self.deep_copy_llm:
            self.llm.cpu()
            del self.llm # free memory
            
        
    def print_log(self, log_str):
        with open(log_str, 'a') as wf:
            wf.write(log_str + '\n')


    def get_instruction_finetune_data(self, init_n_train=1024):
        train_chat_data = []
        for i in range(init_n_train):
            sample = ""
            for s in random.sample(self.success_examples, 1):
                sample += s

            train_chat_data.append(sample)
        return train_chat_data
        
        

    def log2examples(self, log_path):
        success_examples = []
        with open(log_path, 'r') as f:
            example = ''
            for line in f:
                if line.startswith('[INST] [System]: You are playing') and example is not '':
                    success_examples.append(example)
                    example = ''
                example += line
        return success_examples



class ChameleonEnv:
    def __init__(self,
                 llm,
                 tokenizer,
                 num_players: int = 3,
                 max_steps: int = 16,
                 log_path: str = None,
                 seed: int = 1,
                 ) -> None:
        
        
        player_names = []
        self.players = []
        self.buffers = []
        assert num_players <= 6
        for j in range(num_players):
            p = chameleon_example["players"][j]
            player = ChameleonPlayer(name=p["name"], 
                                     role_desc=p["role_desc"],
                                     global_prompt=chameleon_example["global_prompt"], 
                                     backend=llm,
                                     tokenizer=tokenizer,
                                     )
            player_names.append(p["name"])
            self.players.append(player)
        
        # env = Chameleon(player_names, topic_codes=HARD_DEFAULT_TOPIC_CODES)
        self.environment = Chameleon(player_names, topic_codes=HARD_DEFAULT_TOPIC_CODES)

        self.max_steps = max_steps
        self.current_timestep = self.environment.reset()
        self.uuid = uuid.uuid4()  # Generate a unique id for the game
        self.invalid_actions_retry = 5
        self.LOG_PATH = log_path
        random.seed(seed*100 + num_players) 
        self.reset()

    @property
    def num_players(self):
        return self.environment.num_players

    @property
    def name_to_player(self) -> Dict[str, Player]:
        return {player.name: player for player in self.players}

    def reset(self) -> TimeStep:
        # Reset the environment
        self.current_timestep = self.environment.reset()
        # Reset the players
        for player in self.players:
            player.reset()
        # Reset the uuid
        self.uuid = uuid.uuid4()
        return self.current_timestep


    def step(self) -> TimeStep:
        """Take a step in the game: one player takes an action and the environment updates."""
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]  # get the player object
        observation = self.environment.get_observation(
            player_name
        )  # get the observation for the player

        timestep = None
        for i in range(
            self.invalid_actions_retry
        ):  # try to take an action for a few times
            action = player(observation)  # take an action
            if self.environment.check_action(action, player_name):  # action is valid
                timestep = self.environment.step(
                    player_name, action
                )  # update the environment
                break
            else:  # action is invalid
                print(f"{player_name} made an invalid action {action}")
                continue

        if (
            timestep is None
        ):  # if the player made invalid actions for too many times, terminate the game
            warning_msg = f"{player_name} has made invalid actions for {self.invalid_actions_retry} times. Terminating the game."
            print(warning_msg)
            raise TooManyInvalidActions(warning_msg)
        return timestep


    def next_is_human(self):
        """Check if the next player is human."""
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]
        return isinstance(player.backend, Human)

    def run(self, num_steps: int = -1):
        """Run the game for num_turns."""
        if num_steps < 0:
            num_steps = self.max_steps
        for i in range(num_steps):
            timestep = self.step()
            if timestep.terminal:
                for n, r in timestep.reward.items():
                    player = self.name_to_player[n]
                    # player.buffer.rewards[-1] = r * 2. - 1.
                    player.buffer.rewards[-1] = r
                    player.print_scratchpad(self.LOG_PATH)
                    self.buffers.append(player.buffer)
                break


class ChameleonPlayer:
    def __init__(self,
                 name: str,
                 role_desc: str,
                 global_prompt: str,
                 backend,
                 tokenizer,
                 ) -> None:
        
        self.name = name
        self.role_desc = role_desc
        self.global_prompt = global_prompt
        self.backend=backend
        self.tokenizer = tokenizer
        self.CUTOFF_LEN = 4096

        self.reset()

    
    def update_scratchpad(self, text, tokens=None, logprobs=None, state_values=None, mask=1) -> None:
        self.scratchpad += text
        actions = self.tokenizer.encode(text, add_special_tokens=False)

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

    
    def print_scratchpad(self, log_dir=None):
        info = self.scratchpad
        # info = self.scratchpad.replace("<EOS>\n", "<EOS>")
        # info = info.replace("<EOS>", "<EOS>\n")
        # print(info)
        if log_dir is None:
            print(info)
        else:
            with open(log_dir, 'a') as wf:
                wf.write(info  + '\n')

    

    def reset(self) -> None:
        # self.messages = [("System", self.global_prompt), ("System", self.role_desc)]
        # self.system_messages = [("System", self.global_prompt), ("System", self.role_desc)]
        # self.max_tokens = 512
        self.max_tokens = 64
        self.cur_idx = 0
        self.finished = False
        self.scratchpad = ''
        self.buffer = RolloutBuffer()
        # self.update_scratchpad(self.get_cur_message())
            
    @staticmethod
    def _msg_template(agent_name, content):
        return f"[{agent_name}]: {content}"
    
    def __call__(self, observation: List[Message]) -> str:
        return self.query_act(observation)
    
    def query_act(self, observation: List[Message]) -> str:
        """
        Take an action based on the observation (Generate a response), which can later be parsed to actual actions that affect the game dynamics.

        Parameters:
            observation (List[Message]): The messages that the player has observed from the environment.

        Returns:
            str: The action (response) of the player.
        """
        try:
            response = self.query(history_messages=observation)
        except RetryError as e:
            err_msg = f"Agent {self.name} failed to generate a response. Error: {e.last_attempt.exception()}. Sending signal to end the conversation."
            response = SIGNAL_END_OF_CONVERSATION + err_msg

        return response


    def query(
        self,
        history_messages: List[Message],
    ) -> str:
        agent_name = self.name
        role_desc = self.role_desc
        global_prompt = self.global_prompt
        user_inputs, generated_responses = [], []
        all_messages = (
            [(SYSTEM, global_prompt), (SYSTEM, role_desc)]
            if global_prompt
            else [(SYSTEM, role_desc)]
        )

        for msg in history_messages:
            all_messages.append((msg.agent_name, msg.content))

        prev_is_user = False  # Whether the previous message is from the user
        for i, message in enumerate(all_messages):
            if i == 0:
                assert (
                    message[0] == SYSTEM
                )  # The first message should be from the system

            if message[0] != agent_name:
                if not prev_is_user:
                    user_inputs.append(self._msg_template(message[0], message[1]))
                else:
                    user_inputs[-1] += "\n" + self._msg_template(message[0], message[1])
                prev_is_user = True
            else:
                if prev_is_user:
                    generated_responses.append(message[1])
                else:
                    generated_responses[-1] += "\n" + message[1]
                prev_is_user = False

        assert len(user_inputs) == len(generated_responses) + 1
        past_user_inputs = user_inputs[:-1]
        new_user_input = user_inputs[-1]

        # Recreate a conversation object from the history messages
        conversation = Conversation(
            text=new_user_input,
            past_user_inputs=past_user_inputs,
            generated_responses=generated_responses,
        )

        # Get the response
        response = self._get_response(conversation)
        return response
    
    # @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, conversation):
        if hasattr(self.tokenizer, "apply_chat_template"):
            conversation_input_ids = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        # elif hasattr(self.tokenizer, "_build_conversation_input_ids"):
        #     conversation_input_ids = self._build_conversation_input_ids(conversation)
        # elif hasattr(self.tokenizer, "_legacy_parse_and_tokenize"):
        #     conversation_input_ids = self.tokenizer._legacy_parse_and_tokenize(conversation)
        else:
            raise NotImplementedError
        # print(conversation_input_ids[0])
        if conversation_input_ids[0] == 1 and self.cur_idx == 0:
            self.cur_idx += 1
        cur_string = self.tokenizer.decode(conversation_input_ids[self.cur_idx:])
        # cur_string += f"[{self.name}]:"
        # cur_string = full_string[self.cur_idx:]
        
        self.update_scratchpad(cur_string)

        
        
        llm_output = self.predict(self.buffer.actions, 0.0, 1, self.max_tokens)
        self.update_scratchpad(*llm_output, mask=0)
        # self.cur_idx = len(self.buffer.actions) - 1
        self.cur_idx = len(self.buffer.actions) + 0
        # self.cur_idx = len(self.scratchpad)

        if dist.get_rank() == 0:
            # print("===="*20)
            # print('conversation: ', conversation)
            # print("===="*20)
            # print('cur_string: ', cur_string)
            print("===="*20)
            print('scrachpad: ', self.scratchpad)
            print("===="*20)
            print('enc_text : ', self.tokenizer.decode(self.buffer.actions))
            print("===="*20)
        
        response, tokens, logprobs, state_values = llm_output
        return response
    
    def _build_conversation_input_ids(self, conversation: "Conversation") -> List[int]:
        inputs = []
        for is_user, text in conversation.iter_texts():
            if is_user:
                # We need to space prefix as it's being done within blenderbot
                inputs.append(" " + text)
            else:
                # Generated responses should contain them already.
                inputs.append(text)

        full_string = "  ".join(inputs)
        input_ids = self.tokenizer.encode(full_string)
        if len(input_ids) > self.CUTOFF_LEN:
            input_ids = input_ids[-self.CUTOFF_LEN :]
            print(f"Trimmed input from conversation as it was longer than {self.CUTOFF_LEN} tokens.")
        return input_ids
    
    def _legacy_parse_and_tokenize(self, conversation: List[Conversation]) -> List[int]:
        eos_token_id = self.tokenizer.eos_token_id
        input_ids = []
        for is_user, text in conversation.iter_texts():
            input_ids.extend(self.tokenizer.encode(text, add_special_tokens=False) + [eos_token_id])

        if len(input_ids) > self.CUTOFF_LEN:
            input_ids = input_ids[-self.CUTOFF_LEN:]
        return input_ids

    def predict(self,
                actions,
                top_p,
                temperature,
                max_length_tokens):
        
        fake_input_ids = self.tokenizer('fake example', return_tensors="pt")["input_ids"].to(self.backend.device)
        # input_ids = self.tokenizer.encode(new_text)
        input_ids = torch.tensor([actions]).type_as(fake_input_ids)

    
        # stop_words = ["[|Human|]", "[|AI|]"]
        # stop_words = ["\n", "\n\n"]
        stop_words = ["\n", "</s>", "<EOS>"]
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
                    outputs = self.backend(input_ids)
                else:
                    outputs = self.backend(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
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



