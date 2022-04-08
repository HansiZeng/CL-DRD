#!/bin/bash 
cd ../..

python evaluation/continue_rerank_evaluator.py --queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv \
                                    --collection_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/collection.tsv \
                                    --dev_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/runs/dev/tas256-dev-top200.dev.1000.tsv \
                                    --dev_queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.dev.tsv \
                                    --dev_qrels_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/qrels.dev.small.tsv \
                                    --dev_batch_size=128 \
                                    --share_weights \
                                    --resume_folder=/home/hzeng_umass_edu/scratch/my-msmarco-passage/experiments/baselines/experiment_03-28_144043/models/

