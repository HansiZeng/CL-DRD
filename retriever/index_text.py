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
from retriever.retrieval_utils import get_embeddings_from_scratch, write_embeddings_to_memmap
from dataset import SequenceDataset

BLOCK_SIZE = 50_000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_01-29_230126/models/checkpoint_120000.pth.tar")
    parser.add_argument("--model_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("--tokenizer_name_or_path", default="distilbert-base-uncased")
    
    parser.add_argument("--passages_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/collection.tsv")
    parser.add_argument("--max_length", default=256)

    parser.add_argument("--index_dir", default="/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_01-29_230126/index/")
    parser.add_argument("--is_query", default=False)
    parser.add_argument("--is_parallel", default=True)

    parser.add_argument("--share_weights", action="store_true", default=False)
    
    args = parser.parse_args()

    if args.share_weights:
        assert False

    assert args.index_dir[:-7] in args.resume

    if not os.path.exists(args.index_dir):
        os.mkdir(args.index_dir)

    return args

def main(args):
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
    else:
        assert False, args.resume
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    if args.is_query:
        dataset = SequenceDataset.create_from_seqs_file(args.queries_path, tokenizer, args.max_length, is_query=args.is_query)
    else:
        dataset = SequenceDataset.create_from_seqs_file(args.passages_path, tokenizer, args.max_length, is_query=args.is_query)
    text_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    text_embs, text_ids = get_embeddings_from_scratch(model, text_loader, use_fp16=True, is_query=args.is_query, show_progress_bar=True)
    text_id_to_idx = {tid:idx for idx, tid in enumerate(text_ids)}

    print("embs dtype: ", text_embs.dtype)

    index = faiss.IndexFlatIP(text_embs.shape[1])
    index = faiss.IndexIDMap(index)

    assert isinstance(text_ids, list)
    text_ids = np.array(text_ids)

    index.add_with_ids(text_embs, text_ids)

    if args.resume:
        index_path = Path(args.resume).stem.split(".")[0] + ".index"
        index_path = os.path.join(args.index_dir, index_path)
    else:
        raise ValueError("not index path defined.")

    faiss.write_index(index, index_path)

    meta = {"text_ids": text_ids, "text_id_to_idx": text_id_to_idx}
    with open(os.path.join(args.index_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    

if __name__ == "__main__":
    args = get_args()
    main(args)