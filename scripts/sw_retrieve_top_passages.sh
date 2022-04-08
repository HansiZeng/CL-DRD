#!/bin/bash


RESUME_PATH=/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/baselines/experiment_03-28_144043/models/checkpoint_250000.pth.tar
INDEX_PATH=/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/baselines/experiment_03-28_144043/index/checkpoint_250000.index
OUTPUT_PATH=/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/baselines/experiment_03-28_144043/runs/checkpoint_250000

# msmarco-dev
python retriever/retrieve_top_passages.py \
--queries_path=/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/queries.dev.tsv \
--resume=$RESUME_PATH \
--output_path=${OUTPUT_PATH}.dev.run \
--index_path=$INDEX_PATH \
--share_weights
# trec-19
python retriever/retrieve_top_passages.py \
--queries_path=/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/trec-19/msmarco-test2019-queries.tsv \
--resume=$RESUME_PATH \
--index_path=$INDEX_PATH \
--output_path=${OUTPUT_PATH}.trec19.run \
--share_weights
# trec-20
python retriever/retrieve_top_passages.py \
--queries_path=/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/trec-20/msmarco-test2020-queries.tsv \
--resume=$RESUME_PATH \
--index_path=/$INDEX_PATH \
--output_path=${OUTPUT_PATH}.trec20.run \
--share_weights
