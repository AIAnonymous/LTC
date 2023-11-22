import re, string, os, sys
from typing import List, Union, Literal
from enum import Enum
import os.path
import math
import numpy as np
from tqdm import tqdm
from functools import partial
from packaging import version

from transformers import LlamaTokenizer, LlamaForCausalLM
from datetime import datetime, timedelta

from transformers import Trainer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset, DistributedSampler


import time
import json
import random
import pickle
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, RandomSampler
from accelerate import Accelerator
from accelerate import __version__ as accelerate_version
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin
# from app_modules.utils import greedy_search, is_stop_word_or_prefix

# Train
import transformers
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
import json
import jsonlines

import torch.distributed as dist


from envs.communicate_gsm8k_env import GSM8kCoordinator
from envs.communicate_hotpotqa_env import ReactQACoordinator
from envs.communicate_alfworld_env import ReactAlfworldCoordinator
from envs.communicate_chameleon_env import ChameleonCoordinator
from iterative_ppo import PPOTrainer, PPODataset, PPODataCollatorForLanguageModeling, ActorCriticLlama, RolloutBuffer, SupervisedDataset


class PPOIterativePipeline():
    def __init__(self,
                 env_type: str,
                 train_path: str,
                 test_path: str,
                 local_rank: int = 0,
                 world_size: int = 1,
                 output_dir: str = 'checkpoints/temp_cpt/',
                 model_path: str = None,
                 ) -> None:
        self.world_size = world_size
        self.local_rank = local_rank
        self.global_rank = dist.get_rank()
        self.random_seed = self.global_rank + 0
        random.seed(self.random_seed)
        self.OUTPUT_DIR = output_dir
        self.MODEL_PATH = model_path
        self.training_log_path = os.path.join(self.OUTPUT_DIR, 'log', 'loss', f'loss_log_rank{self.global_rank}.txt')
        if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'log', 'loss')):
            os.makedirs(os.path.join(self.OUTPUT_DIR, 'log', 'loss'), exist_ok = True)
        
        self.int8_training = False
        self.env_type = env_type
           

        if self.env_type == 'gsm8k':
            self.env_cls = GSM8kCoordinator
            with jsonlines.open(train_path) as f:
                all_train_json = [l for l in f]
            with jsonlines.open(test_path) as ques_file:
                all_test_json = [l for l in ques_file]
        elif self.env_type == 'hotpotqa':
            self.env_cls = ReactQACoordinator
            with jsonlines.open(train_path) as f:
                all_train_json = [l for l in f]
            with open(test_path) as f:
                all_test_json = json.load(f)[:self.test_size] # for efficiency
        elif self.env_type == 'chameleon':
            self.env_cls = ChameleonCoordinator
            all_train_json = [i for i in range(1024)] 
            all_test_json = [i for i in range(128)]
        elif self.env_type == 'alfworld':
            self.env_cls = ReactAlfworldCoordinator
            all_train_json = [i for i in range(3553)]
            all_test_json = [i for i in range(134)]
        else:
            raise NotImplementedError
        
        self.init_ppo()
        
        print('total train size: ', len(all_train_json))
        assert len(all_train_json) >= self.dev_size
        div = len(all_train_json) // self.world_size
        # use all train data
        remainder = len(all_train_json) % self.world_size
        if remainder == 0:
            self.train_json = all_train_json[self.global_rank*div:(self.global_rank+1)*div]
        elif self.global_rank < remainder:
            div += 1
            self.train_json = all_train_json[self.global_rank*div:(self.global_rank+1)*div]
        else:
            self.train_json = all_train_json[self.global_rank*div+remainder:self.global_rank*div+remainder+div]
        # random.shuffle(self.train_json)
        self.train_envs = self.env_cls(all_train_json[:16], self.model, self.tokenizer, self.OUTPUT_DIR)
        del self.train_envs.llm # free memory
        with torch.no_grad():
            torch.cuda.empty_cache()

        print('total test size: ', len(all_test_json))
        assert len(all_test_json) >= self.test_size
        div = len(all_test_json) // self.world_size
        # use all test data
        remainder = len(all_test_json) % self.world_size
        if remainder == 0:
            self.test_json = all_test_json[self.global_rank*div:self.global_rank*div+div]
        elif self.global_rank < remainder:
            div += 1
            self.test_json = all_test_json[self.global_rank*div:self.global_rank*div+div]
        else:
            self.test_json = all_test_json[self.global_rank*div+remainder:self.global_rank*div+remainder+div]
        
        self.test_envs = self.env_cls(self.test_json, self.model, self.tokenizer, self.OUTPUT_DIR)    
        if self.env_type == 'gsm8k':
            init_instructions = self.train_envs.get_instruction_finetune_data(8192)
            # set it to None if the model has already finetuned
            self.init_finetune_data = SupervisedDataset(init_instructions, self.tokenizer_train) 
        elif self.env_type == 'hotpotqa':
            init_instructions = self.train_envs.get_instruction_finetune_data(4096)
            self.init_finetune_data = SupervisedDataset(init_instructions, self.tokenizer_train)
        elif self.env_type == 'chameleon':
            init_instructions = self.train_envs.get_instruction_finetune_data(2048)
            self.init_finetune_data = SupervisedDataset(init_instructions, self.tokenizer_train)
        elif self.env_type == 'alfworld':
            init_instructions = self.train_envs.get_instruction_finetune_data(2048)
            self.init_finetune_data = SupervisedDataset(init_instructions, self.tokenizer_train)
        else:
            raise NotImplementedError
        

        self.optimizer = None
        self.scheduler = None
        self.optimizer_state_dict = None
        # create accelerator object
        self.create_accelerator()
        

        self.ppo_input_ids = None
        self.ppo_attention_mask = None
        self.ppo_logprobs = None
        self.ppo_rewards = None
        self.ppo_state_values = None
        self.ppo_system_masks = None

    def load_model(self, adapter_model):
        device_map = None
        if self.ddp:
            device_map = {"": self.local_rank}
        if self.int8_training:
            self.model = ActorCriticLlama.from_pretrained(
                        self.MODEL_PATH,
                        load_in_8bit=True,
                        device_map=device_map,
                    )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = ActorCriticLlama.from_pretrained(
                    self.MODEL_PATH,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
        self.model.training_log_path = self.training_log_path
        if adapter_model is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_model,
                # torch_dtype=torch.float16,
                torch_dtype=torch.float32,
            )
        
    
    def init_ac_model(self):
        device_map = None
        if self.ddp:
            device_map = {"": self.local_rank}

        TARGET_MODULES = [
            "q_proj",
            "k_proj",
            "v_proj",
            "down_proj",
            "gate_proj",
            "up_proj",
            "state_value_head",
        ]
        
        if self.int8_training:
            ac_model = ActorCriticLlama.from_pretrained(
                self.MODEL_PATH,
                load_in_8bit=True,
                device_map=device_map,
            )
            ac_model.share_ac_weights()
            ac_model = prepare_model_for_kbit_training(ac_model)
        else:
            ac_model = ActorCriticLlama.from_pretrained(
                self.MODEL_PATH,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
            ac_model.share_ac_weights()
        ac_model.training_log_path = self.training_log_path
       
        
        config = LoraConfig(
            r=self.LORA_R,
            lora_alpha=self.LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=self.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # config.save_pretrained(self.OUTPUT_DIR)
        ac_model = get_peft_model(ac_model, config)
        ac_model.load_state_dict(self.model.state_dict(), strict=True)
        del self.model # free memory
        with torch.no_grad():
            torch.cuda.empty_cache()
        self.model = ac_model
        
        

    def init_ppo(self):

        if self.env_type == 'gsm8k':
            self.max_iter = 16
            self.test_size = 512
            self.dev_size = 4096 
            self.train_size = 4096 * 4
            self.CUTOFF_LEN = 1024 
            self.BATCH_SIZE = 128
            self.LEARNING_RATE = 0.0001 * self.BATCH_SIZE / 32
            self.sample_gamma = 0.9998
            self.test_iters = 1
        elif self.env_type == 'hotpotqa':
            self.max_iter = 8
            # self.test_size = 8
            self.test_size = 500
            self.dev_size = 2048
            self.train_size = 2048 * 8
            # self.train_size = 32
            self.CUTOFF_LEN = 2048 # TODO increase?
            if "13b" in self.MODEL_PATH:
                self.CUTOFF_LEN = 1536 # TODO increase?
            # self.BATCH_SIZE = 128
            self.BATCH_SIZE = 32
            if "meta-llama/Llama-2" in self.MODEL_PATH or  "llama_2" in self.MODEL_PATH:
                self.LEARNING_RATE = 0.0001 * self.BATCH_SIZE / 32
            else:
                self.LEARNING_RATE = 0.0004 * self.BATCH_SIZE / 32
            self.sample_gamma = 0.99996
            self.test_iters = 1
        elif self.env_type == 'chameleon':
            self.max_iter = 16
            self.test_size = 128
            self.dev_size = 64 
            self.train_size = 128
            self.CUTOFF_LEN = 512 
            self.BATCH_SIZE = 32
            self.LEARNING_RATE = 0.00004 * self.BATCH_SIZE / 32
            self.sample_gamma = 0.9992
            self.test_iters = 1
        elif self.env_type == 'alfworld':
            self.max_iter = 16
            self.test_size = 128
            self.dev_size = 256 
            self.train_size = 1024
            self.CUTOFF_LEN = 1024 # TODO increase?
            self.BATCH_SIZE = 32
            self.LEARNING_RATE = 0.0002 * self.BATCH_SIZE / 32
            self.sample_gamma = 0.9992
            self.test_iters = 1
        else:
            raise NotImplementedError
        
        self.local_dev_size = self.dev_size // self.world_size
        self.local_test_size = self.test_size // self.world_size
        self.local_train_size = self.train_size // self.world_size
        
                
        # Training Parameters
        self.MICRO_BATCH_SIZE = 1
        # self.BATCH_SIZE = 128
        
        self.GRADIENT_ACCUMULATION_STEPS = self.BATCH_SIZE // self.MICRO_BATCH_SIZE
        self.INIT_EPOCHS = 3
        self.EPOCHS = 1
        self.weight_decay = 0.001
        
        
        self.LORA_R = 16
        self.LORA_ALPHA = 16
        self.LORA_DROPOUT = 0.05
        self.VAL_SET_SIZE = 0
        self.TARGET_MODULES = [
            "q_proj",
            "k_proj",
            "v_proj",
            "down_proj",
            "gate_proj",
            "up_proj",
            "state_value_head",
        ]
              
        # Load Model
        # world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = self.world_size != 1
        device_map = None
        if self.ddp:
            device_map = {"": self.local_rank}
            self.GRADIENT_ACCUMULATION_STEPS = self.GRADIENT_ACCUMULATION_STEPS // self.world_size

        if 'llama' in self.MODEL_PATH or 'Llama' in self.MODEL_PATH:
            # self.base_model = 'decapoda-research/' + self.model_name.split('-hf')[0] + '-hf'
            # self.model = ActorCriticLlama.from_pretrained(
            #         self.base_model,
            #         # load_in_8bit=True,
            #         torch_dtype=torch.float16,
            #         device_map=device_map,
            #     )
            self.base_model = self.MODEL_PATH
            if self.int8_training:
                self.model = ActorCriticLlama.from_pretrained(
                    self.MODEL_PATH,
                    load_in_8bit=True,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
                self.model.share_ac_weights()
                self.model = prepare_model_for_kbit_training(self.model)
                print('int8 train lm_head shape:', self.model.lm_head.weight.shape)
            else:
                self.model = ActorCriticLlama.from_pretrained(
                    self.MODEL_PATH,
                    # load_in_8bit=True,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                )
                self.model.share_ac_weights()
            # print('original lm_head shape:', self.model.lm_head.weight.shape)
            
            self.model.training_log_path = self.training_log_path
            self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
            self.tokenizer_train = LlamaTokenizer.from_pretrained(self.base_model, add_eos_token=True, model_max_length=self.CUTOFF_LEN, padding_side="right")
            self.data_collator=PPODataCollatorForLanguageModeling(self.tokenizer_train, mlm=False)
            

            config = LoraConfig(
                r=self.LORA_R,
                lora_alpha=self.LORA_ALPHA,
                target_modules=self.TARGET_MODULES,
                lora_dropout=self.LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # config.save_pretrained(self.OUTPUT_DIR)
            self.model = get_peft_model(self.model, config)
            print('peft lm_head shape:', self.model.lm_head.weight.shape)

            

            total_params,params=0,0
            self.tokenizer_train.pad_token_id = 0 
            for n,p in self.model.model.named_parameters():
                if any([x in n for x in ["lora"]]):
                    total_params += p.numel()
                params += p.numel()
                
           
            print('Total number of lora parameters: {}M, rate: {}%'.format(total_params//1000/1000,round(total_params/params*100,2)))
            print('Total number of trainable parameters on rank{}: {}'.format(self.global_rank, sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)))
            
        else:
            raise NotImplementedError
        
        
            
    def async_generate(self, enable_train=False, enable_test=False):
        model_device = self.model.device
        count = 0
        # print(self.model)
        latest_model = os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}')
        if os.path.exists(latest_model):
            print('load latest model from ', latest_model)
            self.load_model(latest_model)
        elif self.init_finetune_data is not None:
            
            print("len init_input_ids: ", self.init_finetune_data.input_ids.shape)
            self.instruction_finetune()

        if enable_test:
            self.test_envs = self.env_cls(self.test_json, self.model, self.tokenizer, self.OUTPUT_DIR)
            self.model.cpu()
            self.test_envs.ppo_trial_generate(count)
            self.model.to(model_device)

        while count < self.max_iter:
            
            self.random_seed += 1 # change seed, or the trainer will use default seed resulting in the same samples for each iter
            random.seed(self.random_seed)
            
            count += 1
            
            latest_data = os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}')
            
                
            latest_data = os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}')
            # if count == 1:
            #     latest_data = self.INIT_DATA_PATH
            if os.path.exists(latest_data):
                print('load latest data from ', latest_data)        
                self.ppo_input_ids = torch.load(os.path.join(latest_data, 'ppo_input_ids.pt'))[..., :self.CUTOFF_LEN]
                self.ppo_attention_mask = torch.load(os.path.join(latest_data, 'ppo_attention_mask.pt'))[..., :self.CUTOFF_LEN]
                self.ppo_logprobs = torch.load(os.path.join(latest_data, 'ppo_logprobs.pt'))[..., :self.CUTOFF_LEN]
                self.ppo_rewards = torch.load(os.path.join(latest_data, 'ppo_rewards.pt'))[..., :self.CUTOFF_LEN]
                self.ppo_state_values = torch.load(os.path.join(latest_data, 'ppo_state_values.pt'))[..., :self.CUTOFF_LEN]
                self.ppo_system_masks = torch.load(os.path.join(latest_data, 'ppo_system_masks.pt'))[..., :self.CUTOFF_LEN]
            else:

                #dist.barrier()
                # Collect new data
                self.train_envs = self.env_cls(random.sample(self.train_json, self.local_dev_size), self.model, self.tokenizer, self.OUTPUT_DIR)
                self.model.cpu()
                self.train_envs.ppo_trial_generate(count, test=False)
                self.model.to(model_device)

                # Ensure all processes have reached this point before proceeding
                dist.barrier()
                now = datetime.now()
                print("Finish generating, Current Time =", now)

                
                
                if not self.train_envs.ppo_buffers:
                    print('PPO buffers is empty!')
                    raise NotImplementedError
                
                new_input_ids = []
                new_attention_mask = []
                new_logprobs = []
                new_rewards = []
                new_state_values = []
                new_system_masks = []
                for buffer in self.train_envs.ppo_buffers:
                    # torch.tensor(new_data_dict['input_ids']).to(self.model.device)
                    data_dict = buffer.get_data(max_length=self.CUTOFF_LEN)
                    new_input_ids.append(data_dict["input_ids"])
                    new_attention_mask.append(data_dict["attention_mask"])
                    new_logprobs.append(data_dict["logprobs"])
                    new_rewards.append(data_dict["rewards"])
                    new_state_values.append(data_dict["state_values"])
                    new_system_masks.append(data_dict["system_masks"])

                new_input_ids = torch.tensor(new_input_ids).to(self.model.device)
                new_attention_mask = torch.tensor(new_attention_mask).to(self.model.device)
                new_logprobs = torch.tensor(new_logprobs).to(self.model.device)
                new_rewards = torch.tensor(new_rewards).to(self.model.device)
                new_state_values = torch.tensor(new_state_values).to(self.model.device)
                new_system_masks = torch.tensor(new_system_masks).to(self.model.device)


                if self.global_rank == 0:
                    gathered_outputs1 = [torch.zeros_like(new_input_ids) for _ in range(self.world_size)]
                    gathered_outputs2 = [torch.zeros_like(new_attention_mask) for _ in range(self.world_size)]
                    gathered_outputs3 = [torch.zeros_like(new_logprobs) for _ in range(self.world_size)]
                    gathered_outputs4 = [torch.zeros_like(new_rewards) for _ in range(self.world_size)]
                    gathered_outputs5 = [torch.zeros_like(new_state_values) for _ in range(self.world_size)]
                    gathered_outputs6 = [torch.zeros_like(new_system_masks) for _ in range(self.world_size)]
                    
                else:
                    gathered_outputs1 = None
                    gathered_outputs2 = None
                    gathered_outputs3 = None
                    gathered_outputs4 = None
                    gathered_outputs5 = None
                    gathered_outputs6 = None

                # Use dist.gather to pull all the generated data to the master node.
                dist.gather(new_input_ids, gather_list=gathered_outputs1, dst=0)
                dist.gather(new_attention_mask, gather_list=gathered_outputs2, dst=0)
                dist.gather(new_logprobs, gather_list=gathered_outputs3, dst=0)
                dist.gather(new_rewards, gather_list=gathered_outputs4, dst=0)
                dist.gather(new_state_values, gather_list=gathered_outputs5, dst=0)
                dist.gather(new_system_masks, gather_list=gathered_outputs6, dst=0)
                new_data_buffer1 = torch.cat([torch.zeros_like(new_input_ids) for _ in range(self.world_size)], dim=0)
                new_data_buffer2 = torch.cat([torch.zeros_like(new_attention_mask) for _ in range(self.world_size)], dim=0)
                new_data_buffer3 = torch.cat([torch.zeros_like(new_logprobs) for _ in range(self.world_size)], dim=0)
                new_data_buffer4 = torch.cat([torch.zeros_like(new_rewards) for _ in range(self.world_size)], dim=0)
                new_data_buffer5 = torch.cat([torch.zeros_like(new_state_values) for _ in range(self.world_size)], dim=0)
                new_data_buffer6 = torch.cat([torch.zeros_like(new_system_masks) for _ in range(self.world_size)], dim=0)

                # Now the master node will have all the generated data and can process it.
                if self.global_rank == 0:
                    # Merge all the generated text data.
                    new_data_buffer1 = torch.cat(gathered_outputs1, dim=0)
                    new_data_buffer2 = torch.cat(gathered_outputs2, dim=0)
                    new_data_buffer3 = torch.cat(gathered_outputs3, dim=0)
                    new_data_buffer4 = torch.cat(gathered_outputs4, dim=0)
                    new_data_buffer5 = torch.cat(gathered_outputs5, dim=0)
                    new_data_buffer6 = torch.cat(gathered_outputs6, dim=0)
                # Use dist.broadcast to send the new data from the master node to all nodes.
                dist.broadcast(new_data_buffer1, src=0)
                dist.broadcast(new_data_buffer2, src=0)
                dist.broadcast(new_data_buffer3, src=0)
                dist.broadcast(new_data_buffer4, src=0)
                dist.broadcast(new_data_buffer5, src=0)
                dist.broadcast(new_data_buffer6, src=0)
            
                if self.ppo_input_ids is None:
                    self.ppo_input_ids = new_data_buffer1.cpu()
                    self.ppo_attention_mask = new_data_buffer2.cpu()
                    self.ppo_logprobs = new_data_buffer3.cpu()
                    self.ppo_rewards = new_data_buffer4.cpu()
                    self.ppo_state_values = new_data_buffer5.cpu()
                    self.ppo_system_masks = new_data_buffer6.cpu()
                else:
                    self.ppo_input_ids = torch.cat([self.ppo_input_ids , new_data_buffer1.cpu()], dim=0)
                    self.ppo_attention_mask = torch.cat([self.ppo_attention_mask , new_data_buffer2.cpu()], dim=0)
                    self.ppo_logprobs = torch.cat([self.ppo_logprobs , new_data_buffer3.cpu()], dim=0)
                    self.ppo_rewards = torch.cat([self.ppo_rewards , new_data_buffer4.cpu()], dim=0)
                    self.ppo_state_values = torch.cat([self.ppo_state_values , new_data_buffer5.cpu()], dim=0)
                    self.ppo_system_masks = torch.cat([self.ppo_system_masks , new_data_buffer6.cpu()], dim=0)

                if self.global_rank == 0:
                    # save the latest data
                    if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}')):
                        os.makedirs(os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}'), exist_ok = True)
                    now = datetime.now()
                    print(f'strat to save the generated data for {count} iter at {now}')
                    torch.save(self.ppo_input_ids, os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}', f'ppo_input_ids.pt'))
                    torch.save(self.ppo_attention_mask, os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}', f'ppo_attention_mask.pt'))
                    torch.save(self.ppo_logprobs, os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}', f'ppo_logprobs.pt'))
                    torch.save(self.ppo_rewards, os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}', f'ppo_rewards.pt'))
                    torch.save(self.ppo_state_values, os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}', f'ppo_state_values.pt'))
                    torch.save(self.ppo_system_masks, os.path.join(self.OUTPUT_DIR, 'ppo_data', f'ppo_data_{count}', f'ppo_system_masks.pt'))
                    print(f'saved the generated data for {count} iter at {now}')
                # time.sleep(5)
        
            
            # Ensure all processes have reached this point before proceeding
            dist.barrier()
            

            latest_model = os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}')
            if os.path.exists(latest_model):
                print('load latest model from ', latest_model)
                self.load_model(latest_model)
            elif enable_train:
                now = datetime.now()
                print(f'iter_{count} rank{self.global_rank} strats training, Current Time = {now}')
                # print('iter ', count, "strat training, Current Time =", now)
                cur_buffer_size = len(self.ppo_input_ids)
                
                gamma = self.sample_gamma
                prob = torch.tensor([gamma**(cur_buffer_size - i) for i in range(cur_buffer_size)])
                if cur_buffer_size >= self.train_size:
                    index = prob.multinomial(num_samples=self.train_size, replacement=False)
                else:
                    # index = prob.multinomial(num_samples=self.train_size, replacement=True) # in case overfit
                    index = prob.multinomial(num_samples=cur_buffer_size, replacement=True) # in case overfit
                # self.ppo_train_data = PPODataset(self.ppo_input_ids[index].to(self.model.device), 
                #                                 self.ppo_attention_mask[index].to(self.model.device),
                #                                 self.ppo_logprobs[index].to(self.model.device),
                #                                 self.ppo_rewards[index].to(self.model.device),
                #                                 self.ppo_state_values[index].to(self.model.device),
                #                                 self.ppo_system_masks[index].to(self.model.device))
                self.ppo_train_data = PPODataset(self.ppo_input_ids[-self.train_size:].to(self.model.device), 
                                                self.ppo_attention_mask[-self.train_size:].to(self.model.device),
                                                self.ppo_logprobs[-self.train_size:].to(self.model.device),
                                                self.ppo_rewards[-self.train_size:].to(self.model.device),
                                                self.ppo_state_values[-self.train_size:].to(self.model.device),
                                                self.ppo_system_masks[-self.train_size:].to(self.model.device))
                # Ensure all processes have reached this point before proceeding
                # dist.barrier()
                print("len ppo_input_ids: ", self.ppo_input_ids.shape)
                self.model.enable_input_require_grads()
                self.ppo_train(count=count)


                del self.ppo_train_data

                if enable_test and count % self.test_iters == 0:
                    self.test_envs = self.env_cls(self.test_json, self.model, self.tokenizer, self.OUTPUT_DIR)
                    self.model.cpu()
                    self.test_envs.ppo_trial_generate(count)
                    self.model.to(model_device)
                    
            
            
    def tokenize(self, prompt):
        result = self.tokenizer_train(
            prompt,
            truncation=True,
            max_length=self.CUTOFF_LEN,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }
            
    def instruction_finetune(self):
        # train_data = self.cur_train_data["train"].shuffle().map(generate_and_tokenize_prompt)
        # Training 
        self.model.train()
        trainer = Trainer(
            model=self.model,
            train_dataset=self.init_finetune_data,
            eval_dataset=None,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.MICRO_BATCH_SIZE,
                per_device_eval_batch_size=self.MICRO_BATCH_SIZE,
                gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
                #warmup_steps=100,
                # warmup_steps=10, # TODO: increase warmup steps?
                warmup_steps=0,
                num_train_epochs=self.INIT_EPOCHS,
                learning_rate=self.LEARNING_RATE,
                fp16=True,
                logging_steps=1,
                evaluation_strategy="steps" if self.VAL_SET_SIZE > 0 else "no",
                save_strategy="steps",
                eval_steps=200000 if self.VAL_SET_SIZE > 0 else None,
                save_steps=200000,
                output_dir=self.OUTPUT_DIR,
                save_total_limit=100,
                seed=self.random_seed,
                load_best_model_at_end=True if self.VAL_SET_SIZE > 0 else False,
                ddp_find_unused_parameters=False if self.ddp else None,
                # ddp_find_unused_parameters=False,
            ),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer_train, mlm=False),
        )
        self.model.config.use_cache = False # silence the warnings. Please re-enable for inference!
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     self.model = torch.compile(self.model)

        trainer.train()
        if self.global_rank == 0:
            if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', 'ppo_ckpt_0')):
                os.makedirs(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', 'ppo_ckpt_0'), exist_ok = True)
            # self.model.save_pretrained(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', 'ppo_ckpt_0'))
            # model = self.model.merge_and_unload()
            # model.save_pretrained(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', 'ppo_ckpt_0'))
            # time.sleep(5)
        self.model.eval()
        self.model.config.use_cache = True # re-enable for inference

            
    def old_ppo_train(self, count):
        self.cur_lr = self.LEARNING_RATE * (self.max_iter - count + 1) / self.max_iter
        # self.cur_lr = self.LEARNING_RATE 
        # if count == 1:
        self.init_ac_model() # reload to avoid NCCL timeout bug
        # Training 
        self.model.train()
        trainer = PPOTrainer(
            model=self.model,
            train_dataset=self.ppo_train_data,
            eval_dataset=None,
            # optimizers=(self.optimizer, None),
            args=transformers.TrainingArguments(
                per_device_train_batch_size=self.MICRO_BATCH_SIZE,
                per_device_eval_batch_size=self.MICRO_BATCH_SIZE,
                gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
                #warmup_steps=100,
                # warmup_steps=10, # TODO: increase warmup steps?
                warmup_steps=0,
                num_train_epochs=self.EPOCHS,
                learning_rate=self.cur_lr,
                lr_scheduler_type="constant",
                fp16=True,
                logging_steps=1,
                evaluation_strategy="steps" if self.VAL_SET_SIZE > 0 else "no",
                save_strategy="steps",
                eval_steps=200 if self.VAL_SET_SIZE > 0 else None,
                save_steps=200,
                output_dir=self.OUTPUT_DIR,
                save_total_limit=1,
                seed=self.random_seed,
                load_best_model_at_end=True if self.VAL_SET_SIZE > 0 else False,
                ddp_find_unused_parameters=False if self.ddp else None,
                remove_unused_columns=False,
            ),
            data_collator=PPODataCollatorForLanguageModeling(self.tokenizer_train, mlm=False),
        )
        self.model.config.use_cache = False # silence the warnings. Please re-enable for inference!

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        
                
        if self.optimizer_state_dict is not None:
            trainer.optimizer_state_dict = self.optimizer_state_dict
        
        
        print('Total number of trainable parameters on rank{}: {}'.format(self.global_rank, sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)))
            
        trainer.train()
        
        self.cur_lr = self.LEARNING_RATE * (self.max_iter - count) / self.max_iter
        if trainer.optimizer is not None:
            # trainer.optimizer.load_state_dict(self.optimizer.state_dict())
            for g in trainer.optimizer.param_groups:
                g['lr'] = self.cur_lr 

        self.optimizer_state_dict = trainer.optimizer.state_dict()
        self.optimizer = trainer.optimizer
        

        if self.global_rank == 0:
            # self.model.save_pretrained(self.OUTPUT_DIR)
            if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}')):
                os.makedirs(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}'), exist_ok = True)
            self.model.save_pretrained(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}'))
       

        self.model.eval()
        self.model.config.use_cache = True # re-enable for inference


    def get_optimizer(self):
        if self.optimizer is None:
            ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
            decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.LEARNING_RATE,
                # betas=(0.9, 0.95),
            )


    def get_lr_scheduler(self):
        if self.scheduler is None:
            
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
            )


    def create_accelerator(self):
        grad_acc_kwargs = {"num_steps": self.GRADIENT_ACCUMULATION_STEPS}
        if version.parse(accelerate_version) > version.parse("0.20.3"):
            grad_acc_kwargs["sync_with_dataloader"] = False
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        # create accelerator object
        self.accelerator = Accelerator(
            deepspeed_plugin=None, gradient_accumulation_plugin=gradient_accumulation_plugin
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.ppo_train_data is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.ppo_train_data
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.MICRO_BATCH_SIZE,
            "collate_fn": data_collator,
            "num_workers": 0,
            "pin_memory": True,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = RandomSampler(train_dataset)
            dataloader_params["drop_last"] = False
            dataloader_params["worker_init_fn"] = seed_worker

        # return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return DataLoader(train_dataset, **dataloader_params)

    def prepare(self):
        (
            self.train_loader,
            self.model,
            self.optimizer,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.train_loader, self.model, self.optimizer, self.scheduler

        ) # TODO why do not use the train_loader here


        # #====================BEGIN GIST CHANGE====================
        # self.eval_loader=self.accelerator.prepare(self.eval_loader)
        #====================END GIST CHANGE======================
        self.optimizer.zero_grad()
        self.global_step = 0
        
        if self.global_step > 0:
            skip_steps = self.global_step * self.gradient_accumulation_steps
            print("Skiped {} steps.".format(skip_steps))
            self.train_loader_skiped = self.accelerator.skip_first_batches(
                self.train_loader, num_batches=skip_steps
            )
        else:
            self.train_loader_skiped = self.train_loader
        # self.accelerator.wait_for_everyone() # TODO this may cause the process stucked


    def train_step(self, batch):
        #with self.accelerator.accumulate(self.model):
        if True:
            out = self.model(**batch)
            #total_loss = out.loss
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            total_loss = out["loss"] if isinstance(out, dict) else out[0]
            losses = {"total_loss": total_loss}
            self.accelerator.backward(total_loss)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        return losses

    # def train(self):
    def ppo_train(self, count):
        # self.cur_lr = self.LEARNING_RATE * (self.max_iter - count + 1) / self.max_iter
        self.cur_lr = self.LEARNING_RATE 

        self.get_optimizer()
        if self.optimizer is not None:
            # trainer.optimizer.load_state_dict(self.optimizer.state_dict())
            for g in self.optimizer.param_groups:
                g['lr'] = self.cur_lr 

        self.get_lr_scheduler()
        self.train_loader = self.get_train_dataloader()
        self.model.config.use_cache = False # silence the warnings. Please re-enable for inference!

        print('prepare ....')
        self.prepare()
        self.start_time = time.time()
        self.epoch = 0
        self.data_step = 0
        # while True:
        while self.epoch < self.EPOCHS:
            # if self.data_step >= self.config["train"]["num_training_steps"]:
            #     break
            if self.epoch == 0:
                train_loader = self.train_loader_skiped
            else:
                train_loader = self.train_loader
            for batch in tqdm(train_loader):
            
                for k, v in batch.items():
                    batch[k] = v.to(self.accelerator.device, non_blocking=True)
                    #print('data batch', k, batch[k])
                    

                self.model.train()
                # train step
                with self.accelerator.accumulate(self.model):
                    losses = self.train_step(batch)
                    if self.accelerator.sync_gradients:
                        self.global_step += 1
                self.data_step += 1
            self.epoch += 1

         
        # self.accelerator.save_state(self.work_dir)
        if hasattr(self.model, 'module'):
            self.model = self.model.module
        if self.global_rank == 0:
            # self.model.save_pretrained(self.OUTPUT_DIR)
            if not os.path.exists(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}')):
                os.makedirs(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}'), exist_ok = True)
            self.model.save_pretrained(os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}'))
            # # TODO load the optimizer
            # if self.optimizer_state_dict is not None:
            #     torch.save(self.optimizer_state_dict, os.path.join(self.OUTPUT_DIR, 'ppo_ckpt', f'ppo_ckpt_{count}', 'optimizer.pth'))
        self.model.config.use_cache = True # Please re-enable for inference!


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)




def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


