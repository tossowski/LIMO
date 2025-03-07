models=("../../../LLaVA-Med-dev/checkpoints/Qwen_7b_finetune_LIMO_only" "Qwen/Qwen2.5-VL-7B-Instruct")
dataset_name=("aime")


for model in "${models[@]}"
do
    for name in "${dataset_name[@]}"
    do
        CUDA_VISIBLE_DEVICES='0' \
            python eval.py \
            --model_name_or_path ${model} \
            --data_name ${name} \
            --prompt_type "qwen-instruct" \
            --temperature 0 \
            --start_idx 0 \
            --end_idx -1 \
            --batch_size 10 \
            --n_sampling 1 \
            --k 1 \
            --split "test" \
            --max_tokens 32768 \
            --seed 0 \
            --top_p 1 \
            --surround_with_messages 
    done
done

