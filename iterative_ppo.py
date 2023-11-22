import re, string, os, sys
from typing import List, Union, Literal
from enum import Enum
import os.path

import json
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.distributions import Categorical
# import torch.autocast as autocast
# import torch.cuda.amp.GradScaler as GradScaler

import transformers
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaPreTrainedModel, LlamaModel
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.data.data_collator import _torch_collate_batch
# from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class PPOCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    state_values: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class PPOTrainer(Trainer):
    optimizer_state_dict = None
    # scaler = GradScaler()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict)
            self.optimizer_state_dict = None
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class PPODataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # print(examples)
        # print('----'*20)
        # print(examples[0])
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
            # if "system_masks" in batch:
            #     labels[batch["system_masks"] == 1] = IGNORE_TOKEN_ID # TODO check
            batch["labels"] = labels
        # print(examples)
        # print('----'*20)
        # print(batch)
        return batch


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        # init
        self.actions = [1]
        self.logprobs = [1.]
        self.rewards = [0.]
        self.state_values = [0.]
        self.system_masks = [1]
    
    def append(self, action, logprob, reward, state_value, system_mask):
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.system_masks.append(system_mask)

    def add_system_sequence(self, actions):
        for a in actions:
            self.actions.append(a)
            self.logprobs.append(1.)
            self.rewards.append(0.)
            self.state_values.append(0.)
            self.system_masks.append(1)

    def get_data(self, max_length):
        actions = []
        logprobs = []
        rewards = []
        state_values = []
        system_masks = []
        for i, mask in enumerate(self.system_masks):
            # valid buffer
            if mask != -1:
                actions.append(self.actions[i])
                logprobs.append(self.logprobs[i])
                rewards.append(self.rewards[i])
                state_values.append(self.state_values[i])
                system_masks.append(self.system_masks[i])
        if len(actions) >= max_length:
            attention_mask = [1] * max_length
            return {
                "input_ids": actions[:max_length],
                "attention_mask": attention_mask,
                "logprobs": logprobs[:max_length],
                "rewards": rewards[:max_length],
                "state_values": state_values[:max_length],
                "system_masks": system_masks[:max_length],
            }
        else:
            # since eos_token_id == pad_token_id == 0
            pad_length = max_length - len(actions)
            attention_mask = [1] * (len(actions)+1) + [0] * (pad_length-1) 
            return {
                "input_ids": actions + [0] * pad_length,
                "attention_mask": attention_mask,
                "logprobs": logprobs + [1.] * pad_length,
                "rewards": rewards + [0.] * pad_length,
                "state_values": state_values + [0.] * pad_length,
                "system_masks": system_masks + [1] * pad_length,
            }


    def clear(self):
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.system_masks[:]



class ActorCriticLlama(LlamaPreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight", "state_value_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.sep_model = False
        # self.sep_model = True

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if self.sep_model:
            self.critic_model = LlamaModel(config)
        self.state_value_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.hidden_value_layer_index = -1 # use ith layer output as value input
        #self.gamma = 0.998 # TODO optimize it for long sequence

        self.gamma = 1.
        self.eps_clip = 0.2
        self.ppo_beta = 0.5
        self.ppo_lambda = 0.5

        # Initialize weights and apply final processing
        self.post_init()
        self.training_log_path = None
    
    def share_ac_weights(self):
        if self.sep_model:
            self.critic_model.load_state_dict(self.model.state_dict())

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            rewards: Optional[torch.FloatTensor] = None,
            old_logprobs: Optional[torch.FloatTensor] = None,
            old_state_values: Optional[torch.FloatTensor] = None,
            system_masks: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, PPOCausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        #output_hidden_states = (
        #    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        #)
        output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            #return_dict=return_dict,
            return_dict=True,
        )


        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        # state_values = self.state_value_head(hidden_states)[:,:,0] 
        if self.sep_model:
            outputs_state_values = self.critic_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            state_values = torch.tanh(self.state_value_head(outputs_state_values[0])[:,:,0])
        else:
            state_values = torch.tanh(self.state_value_head(outputs.hidden_states[self.hidden_value_layer_index])[:,:,0])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)
            # loss += 1e-7 * state_values.mean() # to avoid find_unused_parameters error
            # loss += 0. * state_values.mean() # to avoid find_unused_parameters error
            # print('LOSS1:', loss)
            if rewards is None:
                lm_loss += 0. * state_values.mean() # to avoid find_unused_parameters error
            # print('state_values: ', state_values.mean(), state_values)
            loss = lm_loss

        if rewards is not None:
            assert input_ids.shape == state_values.shape == rewards.shape == old_logprobs.shape == old_state_values.shape == system_masks.shape
            value_mask = system_masks == 0
            policy_mask = system_masks != 1 
            # system_masks
            value_loss_fct = MSELoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_state_values = state_values[..., 1:].contiguous().type_as(logits)
            shift_actions = input_ids[..., 1:].contiguous()
            shift_rewards = rewards[..., 1:].contiguous().type_as(logits)
            shift_old_logprobs = old_logprobs[..., 1:].contiguous().type_as(logits)
            shift_old_state_values = old_state_values[..., 1:].contiguous().type_as(logits)
            shift_value_masks = value_mask[..., 1:].contiguous().type_as(logits)
            shift_policy_masks = policy_mask[..., 1:].contiguous().type_as(logits)

            # Monte Carlo estimate of returns
            
            discounted_reward = 0
            for i in reversed(range(shift_rewards.shape[1])):
                discounted_reward = shift_rewards[:, i] + ((1.-shift_value_masks[:, i])*self.gamma + shift_value_masks[:, i])* discounted_reward
                shift_rewards[:, i] = discounted_reward
            
            # Normalizing the rewards
            shift_rewards = torch.tanh(shift_rewards)
              
            # calculate advantages
            advantages = shift_rewards - shift_old_state_values           
            
            # Finding the ratio (pi_theta / pi_theta__old)
            probs = torch.softmax(shift_logits, dim=-1)
      
            dist = Categorical(probs.to(torch.float32))
            dist_entropy = dist.entropy()
            # logprobs = dist.log_prob(dist.sample())
            logprobs = dist.log_prob(shift_actions)
            ratios = torch.exp(logprobs - shift_old_logprobs).type_as(logits)

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages


            value_loss = value_loss_fct(shift_state_values*shift_value_masks, shift_rewards*shift_value_masks)
            policy_loss = (-torch.min(surr1, surr2)*shift_policy_masks).mean()
            entropy_loss = (- 0.01 * dist_entropy*shift_policy_masks).mean()
           
            ppo_loss = policy_loss + self.ppo_lambda * value_loss + entropy_loss
        
            loss_info = f'LM LOSS: {loss} VALUE LOSS: {value_loss.item()} POLICY LOSS: {policy_loss.item()} ENTROPY LOSS: {entropy_loss.item()}'
            print(loss_info)
            if self.training_log_path is not None:
                with open(self.training_log_path, 'a') as wf:
                    wf.write(loss_info + '\n')
            if value_loss.isnan().sum()>0:
                print(shift_state_values, shift_logits.sum(), torch.mean((shift_state_values-shift_rewards)**2), torch.nanmean((shift_state_values-shift_rewards)**2))
            else:
                if shift_rewards.sum() > 0:
                    loss += self.ppo_beta * ppo_loss
                else:
                    loss = 0.00001 * lm_loss + self.ppo_beta * ppo_loss
            

        if not return_dict:
            output = (logits,state_values,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PPOCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            state_values=state_values,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past



def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:

    # Tokenize conversations
    input_ids = tokenizer(
        sources,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    system_prompt_sign = "(END OF EXAMPLES)" # for HotpotQA prompts
    for conversation, target in zip(sources, targets):
        if system_prompt_sign in conversation:
            system_prompt = conversation[:conversation.find(system_prompt_sign)] + system_prompt_sign
            # "-1" is hardcoded for the Llama tokenizer to make the offset correct.
            system_prompt_len = len(tokenizer(system_prompt).input_ids) - 1
            # Ignore the system instructions
            target[:system_prompt_len] = IGNORE_TOKEN_ID

            for i in range(6):
                obs_sign = f"Observation {i+1}:"
                tho_sign = f"Thought {i+2}:"
                if obs_sign in conversation:
                    cur_conv = conversation[:conversation.find(obs_sign)]
                    obs_len = len(tokenizer(cur_conv).input_ids) + 1
                    if tho_sign in conversation:
                        cur_conv = conversation[:conversation.find(tho_sign)]
                        tho_len = len(tokenizer(cur_conv).input_ids) - 1
                        target[obs_len:tho_len] = IGNORE_TOKEN_ID

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
                               


class IFDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        assert len(input_ids) == len(attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
        }

    def __len__(self):
        return len(self.input_ids)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        data_dict = preprocess(raw_data, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

        
class PPODataset(Dataset):
    def __init__(self, input_ids, attention_mask, logprobs, rewards, state_values, system_masks):
        assert len(input_ids) == len(attention_mask)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.logprobs = logprobs
        self.rewards = rewards
        self.state_values = state_values
        self.system_masks = system_masks

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "old_logprobs": self.logprobs[i],
            "rewards": self.rewards[i],
            "old_state_values": self.state_values[i],
            "system_masks": self.system_masks[i],
        }

    def __len__(self):
        return len(self.input_ids)

