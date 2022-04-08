#!/bin/bash 
cd ../..

 python -m torch.distributed.launch --nproc_per_node=4 trainer/knowledge_distill/kd_nway_listwise.py \
                                     --queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv \
                                    --collection_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/collection.tsv \
                                    --training_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/corase_to_fine_grained/20relT_10neg.train.json \
                                    --dev_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/runs/dev/tas256-dev-top200.dev.1000.tsv \
                                    --dev_queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.dev.tsv \
                                    --dev_qrels_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/qrels.dev.small.tsv \
                                    --experiment_folder=/gypsum/scratch1/hzeng/my-msmarco-passage/experiments/kd_nway_listwise \
                                    --kd_mode=ylabel \
                                    --ylabel_mode=ranknet \
                                    --T=50 \
                                    --lambda_weight=10 \
                                    --model_checkpoint="/gypsum/scratch1/hzeng/my-msmarco-passage/experiments/kd_nway_listwise/experiment_01-19_165925/models/checkpoint_250000.pth.tar" \
                                    --num_train_epochs=2 \
                                    --learning_rate="1e-6" \
                                    --label_mode="2"
