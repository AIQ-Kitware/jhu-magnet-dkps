#!/bin/bash


uv run python -m magnet.evaluation \
    jhu_ta1/cards/jhu_run_predict_query_efficiency.yaml \
    --jobs 8

uv run python -m magnet.evaluation \
    jhu_ta1/cards/jhu_run_predict_query_efficiency_mae.yaml \
    --jobs 8