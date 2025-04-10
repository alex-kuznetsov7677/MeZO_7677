#!/bin/bash


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ALLOW_WEIGHTS_ONLY_LOAD="1"


TASK=${TASK:-"MRPC"}
MODEL=${MODEL:-"roberta-large"}
BS=${BS:-2}
STEPS=${STEP:-1000}
EVAL_STEP=${EVAL_STEP:-100}
LR=${LR:-1e-5}
EXTRA_TAG=${EXTRA_TAG:-"lora"}
SEED=${SEED:-42}
EPS=${EPS:-1e-2}
WD=${WD:-0.02}
TYPE=${TYPE:-"finetune"} 
TRAINER=${TRAINER:-"standard"}
NUM_GPU=${NUM_GPU:-1}
OPT=${OPT:-"adam"}


# --- LoRA params ---
LORA_FLAGS="--apply_lora --lora_r 8 --lora_alpha 16"




GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
TAG=${TAG:-${MODEL}-mezo-${EXTRA_TAG}}


TASK_EXTRA=""
if [ "$TYPE" = "prompt" ]; then
    case $TASK in
        SST-2)
            TEMPLATE='*cls**sent_0*_It_was*mask*.*sep+*'
            MAPPING="{'0':'terrible','1':'great'}"
            ;;
        MRPC)
            TEMPLATE='*cls**sent_0**mask*,*+sentl_1**sep+*'
            MAPPING="{'0':'No','1':'Yes'}"
            ;;
    esac
    PROMPT_ARGS="--template '$TEMPLATE' --mapping '$MAPPING'"
else
    PROMPT_ARGS=""
fi


ALL_ARGS=(
    --no_cuda
    --model_name_or_path "$MODEL" 
    --few_shot_type "$TYPE"
    --task_name "$TASK"
    --data_dir "data/full/MRPC"
    --overwrite_output_dir --output_dir "result/$TASK-$MODEL-$TYPE-$TRAINER-$TAG$GR_TAG/$SEED"
    --tag "$TAG"
    --max_seq_length 128
    --seed "$SEED"
    --do_eval --do_predict --do_train
    --gradient_checkpointing

    --learning_rate "$LR"
    --per_device_train_batch_size "$BS"
    --trainer "$TRAINER"
    --lr_scheduler_type "constant"
    --optimizer "sgd"
    --max_steps "$STEPS"
    --weight_decay "$WD"
    --metric_for_best_model "f1"
    --max_grad_norm 1.

    --evaluation_strategy "steps"
    --evaluate_during_training
    --per_device_eval_batch_size 8
    --eval_steps "$EVAL_STEP"
    --logging_steps 10
    --logging_dir "./logs"
    --log_level "info"

    --zero_order_optim
    --use_zo_grad_est
    --zero_order_sample 1
    --zero_order_eps "$EPS"
    --report_to "tensorboard"

    $PROMPT_ARGS
    $LORA_FLAGS
    "$@"
)


if [[ $NUM_GPU -gt 1 ]]; then
    PORT_ID=$(expr $RANDOM + 1000)
    export OMP_NUM_THREADS=8
    python -m torch.distributed.launch --nproc_per_node "$NUM_GPU" --master_port "$PORT_ID" run.py "${ALL_ARGS[@]}"
else
    python run.py "${ALL_ARGS[@]}"
fi


unset OMP_NUM_THREADS CUDA_VISIBLE_DEVICES PYTORCH_ALLOW_WEIGHTS_ONLY_LOAD
rm -rf "result/$TASK-$MODEL-$TYPE-$TRAINER-$TAG$GR_TAG/$K-$SEED"

