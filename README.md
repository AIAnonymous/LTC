# Adapting LLM Agents Through Communication


This repo contains PyTorch implementation for paper: Adapting LLM Agents Through Communication




## Environment 
You can just run the bash file 'setup.sh' or run the following commands to create a conda environment:
```bash
conda create -n ltc python=3.11
source activate ltc
pip install -r requirements.txt

```


### Train LTC
We fine-tune our models using adapted Hugging Face training code. Below is a command that fine-tunes LLaMA-7B with our dataset on a machine with 8 A100 80G GPUs:
```
model_path=decapoda-research/llama-7b-hf
work_dir=/path/to/workdir
env_type=alfworld
WANDB_DISABLED=true OMP_NUM_THREADS=8 WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 async_generate.py $work_dir $model_path $env_type True


```
This is for Alfworld, and you can also use the bash scripts:
```
bash run_alfworld.sh
```

For HotpotQA, run
```
bash run_hotpotqa.sh
```
For GSM8k, run
```
bash run_gsm8k.sh
```
For Chameleon with ChatArena, run
```
bash run_chameleon.sh
```


