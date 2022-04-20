#!/bin/bash

python artificial_detection/data/proxy.py \
    --df_path="$HOME/atd-data/metrics_checkpoint_merged_df.tsv" \
    --metrics_names="$1" \
    --model_path="$2" \
    --output_path="$HOME/atd-data/collected_metrics_$1.tsv"
