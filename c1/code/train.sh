#!/bin/bash
python -m python.export.export_static_reparam --backbone mobileone_s2 --checkpoint ../kaggle/1296_t672x544/best_model.pth --im_dim 3 --size 672x544 --output ./models/n/best_672x544.onnx --model_type CraterSMP
python -m python.training.train_ranker  --input_data ../ranking_data/features_full.csv --output lightgbm_ranker_n.txt
python -m python.export.convert_ranker_to_c -i ../lightgbm_ranker_n.txt -o ./cpp/include/lightgbm_ranker.h