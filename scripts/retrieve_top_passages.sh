#!/bin/bash


RESUME_PATH=/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_01-29_230126/models/checkpoint_120000.pth.tar
INDEX_PATH=/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_01-29_230126/index/checkpoint_120000.index
OUTPUT_PATH=/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_01-29_230126/cl-drd-runs/checkpoint_120000

# msmarco-dev
python retriever/retrieve_top_passages.py \
--queries_path=/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/queries.dev.tsv \
--resume=$RESUME_PATH \
--output_path=${OUTPUT_PATH}.dev.run \
--index_path=$INDEX_PATH
# trec-19
python retriever/retrieve_top_passages.py \
--queries_path=/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/trec-19/msmarco-test2019-queries.tsv \
--resume=$RESUME_PATH \
--index_path=$INDEX_PATH \
--output_path=${OUTPUT_PATH}.trec19.run
# trec-20
python retriever/retrieve_top_passages.py \
--queries_path=/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/trec-20/msmarco-test2020-queries.tsv \
--resume=$RESUME_PATH \
--index_path=/$INDEX_PATH \
--output_path=${OUTPUT_PATH}.trec20.run
