#! /bin/bash

DATA_PATH="/zhangpai26/100b_training/GLM_100B_data_1/merge.bin"
NAME="UniGLM-811-true-2"
LOAD_DIR="/zhangpai26/uniglm_ckpt/${NAME}"

EXP_NAME=${NAME}-${TIMESTAMP}
CHECKPOINT_PATH="/zhangpai26/uniglm_ckpt/${NAME}"
TENSORBOARD_PATH="runs/uniglm/${NAME}"

config_json="./ds-configs/${EXP_NAME}/ds_config.json"

MICRO_BATCH_SIZE=1
# GLOBAL_BATCH_SIZE=5632 # 176*96/3
# GLOBAL_BATCH_SIZE=3696 # 176*63/3
# GLOBAL_BATCH_SIZE=2816 # 176*48/3
GLOBAL_BATCH_SIZE=8448 # 176*96/4

# TP_SIZE=1
# PP_SIZE=1
# NHIDDEN=640
# FFN_HIDDEN=$((NHIDDEN * 8 / 3))
# NLAYERS=30
# NHEADS=20

TP_SIZE=8
PP_SIZE=1
NHIDDEN=10240
FFN_HIDDEN=$((NHIDDEN * 9 / 3 + 1024)) 
NLAYERS=72
NHEADS=80

# NHIDDEN=10240
# FFN_HIDDEN=$((NHIDDEN * 9 / 3))
# NLAYERS=72
# NHEADS=80

# NHIDDEN=12288
# FFN_HIDDEN=$((NHIDDEN * 8 / 3))
# NLAYERS=56
# NHEADS=64
SPAN_LEN_PER_SAMPLE=1024 # sequence length per sample from BinaryDataset
LENGTH_PER_SAMPLE=1024 # sequence length per sample from BinaryDataset
SEQ_LEN=1024 # actual length during training (pad to this)

# LENGTH_PER_SAMPLE=2000 # sequence length per sample from BinaryDataset
# SEQ_LEN=2048 # actual length during training (pad to this)

SAVE_INTERVAL=300
EVAL_INTERVAL=300

TRAIN_TOKENS=1000000000000 # 450B tokens
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))  # Decay for the first 90% tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 30 / 1000))  # 2.5% warmup
BATCH_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 25 / 1000))  # 2.5% warmup

ZERO_STAGE=1

script_path="pretrain_glm.py"

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 4e-5 \
    --min-lr 4e-6 \
    --override-lr-scheduler \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "
# --lr-auto-warmup-steps 19500 500 \

OUTPUT_ARGS=" \
    --log-interval 1 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 3 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "
    # --load $LOAD_DIR
GLM_ARGS="
       --tensor-model-parallel-size $TP_SIZE \
       --pipeline-model-parallel-size $PP_SIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --num-attention-heads $NHEADS \
       --make-vocab-size-divisible-by 128 \
       --glm \
       --bert-prob 1.0 \
       --span-prob 0.0 \
       --single-span-prob 0.02 \
       --short-seq-prob 0.02 \
       --mask-prob 0.15 \
       --average-block-length 3 \
       --min-gmask-ratio 0.2 \
       --aggregated-samples-per-sequence 1 \
       --deepnorm \
       --position-embedding-type rotary \
       --rotary-embedding-2d \
       --ffn-hidden-size $FFN_HIDDEN \
       --glu-activation geglu \
       --no-bias-gelu-fusion \
       --tokenizer-type ProteinTokenizer \
       --pad-vocab-size-to 128 \
       --attention-dropout 0  \
       --hidden-dropout 0 \
    "

DEEPSPEED_ARGS=" \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --zero-stage $ZERO_STAGE \
       --partition-activations \
       --deepspeed-activation-checkpointing \
    "

gpt_options=" \
       $GLM_ARGS \
       --pp-partition-method 'type:transformer' \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --rampup-batch-size 240 24 $BATCH_WARMUP_SAMPLES \
       --train-samples $TRAIN_SAMPLES \
       --length-per-sample $LENGTH_PER_SAMPLE \
       --seq-length $SEQ_LEN \
       --span-len-per-sample $SPAN_LEN_PER_SAMPLE \
       --multitask-ratio 0.00 \
       --num-workers 1 \
       --data-path $DATA_PATH \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --abort-on-unmet-fused-kernel-constraints \
       --split 949,50,1 \
       --distributed-backend nccl \
       --checkpoint-activations \
       --init-method-std 0.0052 \
       --shrink-logit-embedding-gradient \
       --shrink-embedding-gradient-alpha 0.1 \
       --fp16 \
       $OPTIMIZER_ARGS \
       $DEEPSPEED_ARGS \
       $OUTPUT_ARGS
       
"
# --skip-train-iteration-range 2500-3000 3700-3800\
# --rampup-batch-size 192 32 $BATCH_WARMUP_SAMPLES \
# --pp-partition-method 'type:transformer|embedding' \
#        --rampup-batch-size 210 30 $BATCH_WARMUP_SAMPLES \
    #    --skip-train-iteration-range 40701-40900 42401-42600 \
#       --multitask-data-transform-steps 22850 2000 \
#       --shrink-embedding-gradient-steps 22850 6000 \
#    --multitask-data-path $MULTITASK_DATA_PATH \

mkdir -p ds-configs/${EXP_NAME}
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 200,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "steps_per_print": 1,
  "wall_clock_breakdown": true
}
EOT
