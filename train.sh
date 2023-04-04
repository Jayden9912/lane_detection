python trainer.py \
--exp_dir ./experiments/exp_TU_testing \
--batch_size 8 \
--flip_prob 0.5 \
--sr_ratio 2 2 1 1 \
--device_id 1 \
--se_layer


# --max_epochs 50 \
# --device_id 1

# # CULane
# python trainer.py \
# --exp_dir ./experiments/exp_CU_baseline_flip_sr2211_GELU_se \
# --dataset_name CULane \
# --resize_shape 800 288 \
# --batch_size 4 \
# --flip_prob 0.5 \
# --sr_ratio 2 2 1 1 \
# --se_layer \
# --learning_rate 0.00001 \
# --max_epochs 50 \
# --resume
# # --weight_decay 0.1 \
# # --max_epochs 50 \
# # --device_id 1
# # --attention_drop_rate 0.2 \
