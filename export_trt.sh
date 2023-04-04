python export_trt.py \
--exp_dir ./experiments/exp_TU_baseline_flip_sr2211_GELU_se \
--saving_path ./trt/ \
--dataset Tusimple \
--batch_size 1 \
--toonnx \
--se_layer \
--sr_ratio 2 2 1 1