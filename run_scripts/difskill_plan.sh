python core/diffskill/train_diffskill_rgb.py \
    --run_plan True \
    --resume_path '<PATH TO TRAINED DIFFSKILL AGENT>' \
    --adam_iter 10 \
    --adam_sample 1280 \
    --plan_bs 128 \
    --plan_step 3