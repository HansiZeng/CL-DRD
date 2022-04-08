import sys 
sys.path += ["./"]
import os 
import pickle 
import argparse
from typing import Optional, Union, List, Dict
import glob
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
import time
import random
import yaml
from pathlib import Path
import shutil

import faiss
import torch
import numpy as np 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from models.dual_encoder import DualEncoder 
from retriever.retrieval_utils import get_embeddings_from_scratch, convert_index_to_gpu, index_retrieve
from dataset import SequenceDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    parser.add_argument("--model_name_or_path", default="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
    parser.add_argument("--tokenizer_name_or_path", default="distilbert-base-uncased")
    #parser.add_argument("--passages_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/processed/matched_passages.train.tsv")
    parser.add_argument("--passages_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/rank_queries/passages.dev.small.tsv")
    parser.add_argument("--index_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/index/query/distilbert-dot-tas_b-b256-msmarco.index")

    parser.add_argument("--max_length", default=256)
    parser.add_argument("--top_k", default=200)

    parser.add_argument("--output_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/runs/tas256-dev-small-top200.query.run")

    args = parser.parse_args()

    return args 

def main(args):
    model = DualEncoder(args.model_name_or_path, share_weights=True)
    if args.resume:
        print(f"load model from ==> {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    dataset = SequenceDataset.create_from_seqs_file(args.passages_path, tokenizer, args.max_length, is_query=False)
    passage_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    passage_embs, passage_ids = get_embeddings_from_scratch(model, passage_loader, use_fp16=True, is_query=False, show_progress_bar=True)

    index = faiss.read_index(args.index_path)
    index = convert_index_to_gpu(index, 0, False)

    nn_scores, nn_query_ids = index_retrieve(index, passage_embs, args.top_k, batch=128)

    pid_to_ranks = {}
    for pid, qids, scores in zip(passage_ids, nn_query_ids, nn_scores):
        for qid, s in zip(qids, scores):
            if pid not in pid_to_ranks:
                pid_to_ranks[pid] = [(qid, s)]
            else:
                pid_to_ranks[pid] += [(qid, s)]
    print(f"# unique passages = {len(pid_to_ranks)}")

    total_rank = 0
    with open(args.output_path, "w") as f:
        for pid in pid_to_ranks:
            ranks = pid_to_ranks[pid]
            for i, (qid, s) in enumerate(ranks):
                f.write(f"{pid}\t{qid}\t{i+1}\t{s}\n")
            total_rank += len(ranks)
    
    print(f"average ranks per query = {total_rank/len(pid_to_ranks)}")


if __name__ == "__main__":
    args = get_args()
    main(args)