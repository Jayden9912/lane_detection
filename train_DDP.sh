torchrun --standalone --nproc_per_node=2 trainer_DDP_v3.py \
--exp_dir ./experiments/exp_TU_testing \
--batch_size 4 \
--flip_prob 0.5 \
--sr_ratio 2 2 1 1 \
--se_layer
# --weight_decay 0.1 \
# --max_epochs 50 \
# --device_id 1
# --attention_drop_rate 0.2 \

# # CULane
# python trainer.py \
# --exp_dir ./experiments/exp_CU_baseline_flip_sr2211_GELU_se \
# --dataset_name CULane \
# --resize_shape 800 288 \
# --batch_size 4 \
# --flip_prob 0.5 \
# --sr_ratio 2 2 1 1 \
# --se_layer
# # --weight_decay 0.1 \
# # --max_epochs 50 \
# # --device_id 1
# # --attention_drop_rate 0.2 \
