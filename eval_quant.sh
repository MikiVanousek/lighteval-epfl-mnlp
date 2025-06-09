lighteval accelerate \
    --eval-mode "lighteval" \
    --save-details \
    --override-batch-size 32 \
    --custom-tasks "community_tasks/mnlp_mcqa_evals.py" \
    --output-dir "../eval_output/" \
    ../model_configs/quantized_model.yaml \
    "community|mnlp_mcqa_evals|0|0"

