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
from collections import OrderedDict

import faiss
import torch
import numpy as np 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from models.nway_dual_encoder import NwayDualEncoder
from retriever.retrieval_utils import get_embeddings_from_scratch, convert_index_to_gpu, index_retrieve
from dataset import SequenceDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="")
    parser.add_argument("--model_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("--tokenizer_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("--queries_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/queries.dev.small.tsv")
    parser.add_argument("--index_path", default="")

    parser.add_argument("--max_length", default=30)
    parser.add_argument("--top_k", default=1000)
    parser.add_argument("--is_parallel", default=True)
    parser.add_argument("--share_weights", action="store_true", default=False)

    parser.add_argument("--output_path", default="")

    args = parser.parse_args()

    return args 

def main(args):
    if "train" in args.queries_path:
        print("retrieve train queries")
        assert "train" in args.output_path
    if "dev" in args.queries_path:
        print("retrieve dev queries")
        assert "dev" in args.output_path
    if "2019" in args.queries_path:
        print("retrieve trec-19 queries")
        assert "trec19" in args.output_path
    if "2020" in args.queries_path:
        print("retrieve trec-20 queries")
        assert "trec20" in args.output_path

    model = NwayDualEncoder(args.model_name_or_path, share_weights=args.share_weights)
    print("************************* share weights = {} *************************".format(args.share_weights))
    if args.resume:
        print(f"load model from ==> {args.resume}")
        checkpoint = torch.load(args.resume)
        if args.is_parallel:
            print("load parallel wrapped model.")
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    dataset = SequenceDataset.create_from_seqs_file(args.queries_path, tokenizer, args.max_length, is_query=True)
    query_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    query_embs, query_ids = get_embeddings_from_scratch(model, query_loader, use_fp16=True, is_query=True, show_progress_bar=True)

    index = faiss.read_index(args.index_path)
    index = convert_index_to_gpu(index, 0, False)

    nn_scores, nn_doc_ids = index_retrieve(index, query_embs, args.top_k, batch=128)

    qid_to_ranks = {}
    for qid, docids, scores in zip(query_ids, nn_doc_ids, nn_scores):
        for docid, s in zip(docids, scores):
            if qid not in qid_to_ranks:
                qid_to_ranks[qid] = [(docid, s)]
            else:
                qid_to_ranks[qid] += [(docid, s)]
    print(f"# unique query = {len(qid_to_ranks)}")

    if not os.path.exists(Path(args.output_path).parent):
        os.mkdir(Path(args.output_path).parent)
    total_rank = 0
    with open(args.output_path, "w") as f:
        for qid in qid_to_ranks:
            ranks = qid_to_ranks[qid]
            for i, (docid, s) in enumerate(ranks):
                f.write(f"{qid}\t{docid}\t{i+1}\t{s}\n")
            total_rank += len(ranks)
    
    print(f"average ranks per query = {total_rank/len(qid_to_ranks)}")


if __name__ == "__main__":
    args = get_args()
    main(args)