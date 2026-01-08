#!/bin/bash
./test --raw-dir $1 --model models/n/best_672x544.onnx  --input-res 1296 --tile-size 672x544 --solution-out sol13-test-672.csv --limit-craters 13  --no-ranker --all
#./test --raw-dir $1 --model models/n/best_672x544.onnx  --input-res 1296 --tile-size 672x544 --solution-out valsol12-672.csv --limit-craters 12
