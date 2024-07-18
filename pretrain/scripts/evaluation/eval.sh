#! /bin/bash
# bash scripts/evaluation/eval.sh configs/glm-100b/eval_config.sh
WORLD_SIZE=8
OPTIONS_NCCL="NCCL_DEBUG=VERSION NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

source $1 # model

CHECKPOINT_PATH="/ckpt/global_stepxxx"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./evaluation/main.py \
       --seed 1234 \
       --tokenizer-type ProteinTokenizer \
       --load ${CHECKPOINT_PATH} \
       --temperature 1.0 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --log-interval 1 \
       --no-load-rng \
       --fp16 \
       ${GLM_ARGS}
    #    
    #    --load ${CHECKPOINT_PATH} \
