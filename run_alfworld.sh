


# model_path=decapoda-research/llama-7b-hf # 404
model_path=meta-llama/Llama-2-7b-hf
work_dir=/path/to/workdir
env_type=alfworld
WANDB_DISABLED=true OMP_NUM_THREADS=8 WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 async_generate.py $work_dir $model_path $env_type True

