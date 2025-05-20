for SEED in 78 23 61;
do python train_tacred.py \
  --model_name_or_path distilroberta-base --input_format typed_entity_marker --seed $SEED \
  --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 \
  --run_name distilroberta-base;
done;