#!/bin/bash 
cd ../..

 python -m torch.distributed.launch --nproc_per_node=4 trainer/ctof_grained/nway_listwise_1.py \
                                    --num_train_epochs=4 \
                                    --queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv \
                                    --collection_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/collection.tsv \
                                    --training_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/corase_to_fine_grained/20relT_10neg.train.json \
                                    --dev_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/runs/dev/tas256-dev-top200.dev.1000.tsv \
                                    --dev_queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.dev.tsv \
                                    --dev_qrels_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/qrels.dev.small.tsv \
                                    --experiment_folder=/home/hzeng_umass_edu/scratch/my-msmarco-passage/experiments/multistep-curriculum \
                                    --label_mode=10
                        
