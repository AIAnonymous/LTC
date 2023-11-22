import sys, os
from typing import Tuple
import torch
from pathlib import Path

import torch.distributed as dist
from datetime import datetime, timedelta



from ppo_pipeline import PPOIterativePipeline



def setup_model_parallel() -> Tuple[int, int]:                            
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        return 0, 1

    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200))
    global_rank = dist.get_rank()

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

local_rank, world_size = setup_model_parallel()

output_dir = sys.argv[1]
model_path = sys.argv[2]
env_type = sys.argv[3] # 'hotpotqa' 'gsm8k' 'alfworld' 'chameleon'
enable_test = True if sys.argv[4] == "True" else False

if 'hotpotqa' in env_type:
    train_path = 'data/hotpot_train_v1.1_simplified.json'
    test_path = 'data/hotpot_dev_v1_simplified.json'
elif 'gsm8k' in env_type:
    train_path = 'data/gsm8k_train.jsonl'
    test_path = 'data/gsm8k_test.jsonl'
elif 'alfworld' in env_type:
    train_path = 'data/alfworld/txt003_trial_0.log' # dummy
    test_path = 'data/alfworld/txt003_trial_0.log' # dummy
elif 'chameleon' in env_type:
    train_path = 'data/chameleon/example128_0.log' # dummy
    test_path = 'data/chameleon/example128_0.log' # dummy
else:
    print(model_path)
    raise NotImplementedError
        
iter_agent = PPOIterativePipeline(env_type=env_type,
                                  train_path=train_path, 
                                test_path=test_path, 
                                local_rank=local_rank,
                                world_size=world_size,
                                output_dir=output_dir,
                                model_path=model_path,
                        )
iter_agent.async_generate(enable_train=True, enable_test=enable_test)



