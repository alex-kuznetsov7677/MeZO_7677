#!/bin/bash


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_ALLOW_WEIGHTS_ONLY_LOAD="1"


TASK=${TASK:-"MRPC"}
MODEL=${MODEL:-"roberta-large"}
BS=${BS:-4}
STEP=${STEP:-1000}
EVAL_STEP=${EVAL_STEP:-100}
LR=${LR:-1e-4}
EXTRA_TAG=${EXTRA_TAG:-"lora"}
SEED=${SEED:-42}
TYPE=${TYPE:-"finetune"}  # Переопределяется из run_cpu.sh
TRAINER=${TRAINER:-"standard"}
NUM_GPU=${NUM_GPU:-0}   # Для CPU: 0 GPU
OPT=${OPT:-"adam"}



# LoRA
LORA_FLAGS="--apply_lora --lora_r 4 --lora_alpha 8"

GR_TAG=seed$SEED-bs$BS-lr$LR-step$STEP-evalstep$EVAL_STEP
TAG=${TAG:-k${K}-${MODEL}-${EXTRA_TAG}}

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
    --model_name_or_path "$MODEL" --few_shot_type "$TYPE"
    --task_name "$TASK"
    --data_dir "data/full/MRPC"
    --overwrite_output_dir --output_dir "result/$TASK-$MODEL-$TYPE-$TRAINER-$TAG$GR_TAG/$K-$SEED"
    --tag "$TAG"
    --seed "$SEED"
    --max_seq_length 128

    --do_eval --do_predict --do_train
    --trainer "$TRAINER"
    --optimizer "$OPT"
    --max_steps "$STEP"
    --gradient_accumulation_steps 2
    --per_device_train_batch_size "$BS"

    --evaluate_during_training
    --log_level "info"
    --eval_steps "$EVAL_STEP"
    --logging_steps 10
    --per_device_eval_batch_size 8
    --report_to "tensorboard"
    --logging_dir "./my_logs"

    $PROMPT_ARGS
    $LORA_FLAGS
    "$@"
)


if [[ $NUM_GPU -gt 1 ]]; then
    PORT_ID=$(expr $RANDOM + 1000)
    export OMP_NUM_THREADS=8
    python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID run.py "${ALL_ARGS[@]}"
else
    python run.py "${ALL_ARGS[@]}"
fi


unset OMP_NUM_THREADS CUDA_VISIBLE_DEVICES PYTORCH_ALLOW_WEIGHTS_ONLY_LOAD
rm -rf "result/$TASK-$MODEL-$TYPE-$TRAINER-$TAG$GR_TAG/$K-$SEED"
