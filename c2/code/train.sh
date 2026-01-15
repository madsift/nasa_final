#!/bin/bash
#python -m python.data.prepare_data --input_dir $1 --output_dir $1/train_patches --gt_csv $1/train-gt.csv --target_size 1620 --tilesize 832 640
python train_smp_multi.py \
    --backbone 'mobileone_s0' --data_dir $1/train_patches  --tile_size '832 640' \
    --model CraterSMP \
    --gpus 4 \
    --outdir $1 \
    --batch_size 16 --num_epochs 32 --sigma_clamp '-2.0,2.0'
#python -m python.export.export_static_reparam --checkpoint ./models/best_model.pth --output ./models/best_672x544_fp32.onnx --dtype fp32 --optimize_arm64
#python -m python.training.train_ranker  --input_data ../ranking_data/features_full.csv --output lightgbm_ranker_n.txt
#python -m python.export.convert_ranker_to_c -i ../lightgbm_ranker_n.txt -o ./cpp/include/lightgbm_ranker.h