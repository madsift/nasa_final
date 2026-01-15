#!/bin/bash

python -m python.export.export_static_reparam --checkpoint ./models/best_model.pth --output ./models/best_672x544_fp32.onnx --dtype fp32 --optimize_arm64
python -m python.training.train_ranker  --input_data ../ranking_data/features_full.csv --output lightgbm_ranker_n.txt
python -m python.export.convert_ranker_to_c -i ../lightgbm_ranker_n.txt -o ./cpp/include/lightgbm_ranker.h