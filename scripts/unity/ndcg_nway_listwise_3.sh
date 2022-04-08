#!/bin/bash 
cd ../..

 python -m torch.distributed.launch --nproc_per_node=4 trainer/knowledge_distill/ndcg_nway_listwise_2.py \
                                    --num_train_epochs=2 \
                                    --queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv \
                                    --collection_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/collection.tsv \
                                    --training_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/corase_to_fine_grained/teacher_scores/30T_score.train.json \
                                    --dev_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/runs/dev/tas256-dev-top200.dev.1000.tsv \
                                    --dev_queries_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/queries.dev.tsv \
                                    --dev_qrels_path=/gypsum/scratch1/hzeng/datasets/msmarco-passage/qrels.dev.small.tsv \
                                    --experiment_folder=/home/hzeng_umass_edu/scratch/my-msmarco-passage/experiments/ndcg_nway_listwise \
                                    --neg_score_mode=mean \
                                    --weighing_scheme=ndcgLoss1_scheme \
                                    --label_mode=3 \
                                    --learning_rate=1e-5 \
                                    --model_checkpoint=/gypsum/scratch1/hzeng/my-msmarco-passage/experiments/ndcg_nway_listwise/experiment_01-28_031601/models/checkpoint_130000.pth.tar