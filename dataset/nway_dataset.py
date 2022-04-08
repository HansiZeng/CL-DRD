import os 
import sys 
sys.path += ["./"]
from pathlib import Path 
from typing import Dict, List, Tuple
import logging
logger = logging.getLogger(__name__)
import time
import ujson

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset 

from .utils import load_queries, load_passages

class NwayDataset(torch.utils.data.Dataset):
    def __init__(self, qid_to_query, pid_to_passage, train_examples, 
                        tokenizer, max_query_len, max_passage_len, label_mode):
        super(NwayDataset, self).__init__()
        self.qid_to_query = qid_to_query
        self.pid_to_passage = pid_to_passage
        self.train_examples = train_examples
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len 
        self.label_mode = label_mode

        assert self.label_mode in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        logger.info(f"label mode: {self.label_mode}")
    
    def __getitem__(self, idx):
        example = self.train_examples[idx]
        qid = example["qid"]
        relT_pids = example["relT_pids"]
        neg_pids = example["neg_pids"] 

        query = self.qid_to_query[qid]
        relT_passages = [self.pid_to_passage[pid] for pid in relT_pids]
        neg_passages = [self.pid_to_passage[pid] for pid in neg_pids]
        if self.label_mode == "1":
            assert len(relT_pids) == 1 and len(neg_pids) == 5
            labels = [1.] + [0.] * len(neg_pids)
        elif self.label_mode == "2":
            assert len(relT_pids) == 10 and len(neg_pids) == 20
            labels = [1.] * len(relT_pids) + [1/2.] * 10 + [0.] * 10
        elif self.label_mode == "3":
            assert len(relT_pids) == 10 and len(neg_pids) == 20
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [0.] * len(neg_pids)
        elif self.label_mode == "4":
            assert len(relT_pids) == 10 and len(neg_pids) == 20
            labels = [1.] + [0.9] * 9 + [1/2.] * 10 + [0.] * 10
        elif self.label_mode == "5":
            assert len(relT_pids) == 20 and len(neg_pids) == 10
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [0.] * len(neg_pids)
        elif self.label_mode == "6":
            assert len(relT_pids) == 30 and len(neg_pids) == 0
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [0.] * len(neg_pids)
        elif self.label_mode == "7":
            assert len(relT_pids) == 5 and len(neg_pids) == 25
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [0.] * len(neg_pids) 
        elif self.label_mode == "8":
            assert len(relT_pids) == 5 and len(neg_pids) == 25
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [-0.25] * 12 + [-0.5] * 13
        elif self.label_mode == "9":
            assert len(relT_pids) == 10 and len(neg_pids) == 20
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [-0.25] * 10 + [-0.5] * 10
        elif self.label_mode == "10":
            assert len(relT_pids) == 20 and len(neg_pids) == 10
            labels = list(1. / np.arange(1, 1+len(relT_pids))) + [-0.25] * 5 + [-0.5] * 5
        else:
            raise ValueError(f"{self.label_mode} do not defined")

        return {
            "qid": qid,
            "relT_pids": relT_pids,
            "neg_pids": neg_pids,
            "query": query,
            "relT_passages": relT_passages,
            "neg_passages": neg_passages,
            "labels": labels,
        }

    def __len__(self):
        return len(self.train_examples)

    def collate_fn(self, batch):
        qids, relT_pids, neg_pids, queries, nway_passages, labels = [], [], [], [], [], []
        for idx, elem in enumerate(batch):
            qids.append(elem["qid"])
            relT_pids.append(elem["relT_pids"])
            neg_pids.append(elem["neg_pids"])
            queries.append(elem["query"])
            nway_passages +=  elem["relT_passages"] + elem["neg_passages"]
            labels.append(elem["labels"])

        qids =  np.array(qids, dtype=np.int64)
        relT_pids = np.array(relT_pids, dtype=np.int64)
        neg_pids = np.array(neg_pids, dtype=np.int64)
        bz, relT_len, neg_len = len(batch), relT_pids.shape[1], neg_pids.shape[1]
        nway = relT_len + neg_len

        queries = self.tokenizer(queries, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_query_len)
        nway_passages = self.tokenizer(nway_passages, padding=True, truncation='longest_first', 
                                        return_tensors="pt", max_length=self.max_passage_len)
        nway_passages = {k:v.view(bz, nway, -1) for k, v in nway_passages.items()}
        labels = torch.FloatTensor(labels)
        
        return {
            "qid": qids,
            "relT_pids": relT_pids,
            "neg_pids": neg_pids,
            "nway_pids": np.concatenate((relT_pids, neg_pids), axis=-1),
            "query": queries,
            "nway_passages": nway_passages,
            "labels": labels
        }            

    @classmethod
    def create_from_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, label_mode="3"):
        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))

        train_examples = {}
        with open(training_path, "r") as fin:
            train_examples = ujson.load(fin)
        
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len, label_mode=label_mode)

    @classmethod
    def dist_create_from_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len,
                                rank, nranks):
        assert rank in range(nranks) and nranks > 1
        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))

        train_examples = []
        with open(training_path, "r") as fin:
            for line_idx, line in enumerate(fin):
                if line_idx % nranks == rank:
                    train_examples.append(ujson.loads(line))
        
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len)

    @classmethod
    def create_from_json_line_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, 
                                    label_mode):
        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))
        
        train_examples = []
        with open(training_path, "r") as fin:
            for line in fin:
                example = ujson.loads(line)
                assert "relT_pids" not in example and "rel_pid" in example 
                example["relT_pids"] = [example.pop("rel_pid")]
                train_examples.append(example)

        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len, label_mode)

    @classmethod
    def create_from_relT_most_semi_hard_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, 
                                    label_mode, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1

        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))

        if rank == -1:
            train_examples = []
            with open(training_path, "r") as fin:
                for line in fin:
                    new_example = {}
                    example = ujson.loads(line)
                    train_examples.append({
                        "qid": example["qid"],
                        "relT_pids": example["relT_pids"],
                        "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"],
                    })
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        example = ujson.loads(line)
                        train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len, label_mode)
    
    @classmethod
    def create_from_10relT_20neg_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, 
                                    label_mode, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        assert label_mode in ["3", "9"]

        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))

        if rank == -1:
            train_examples = []
            with open(training_path, "r") as fin:
                for line in fin:
                    new_example = {}
                    example = ujson.loads(line)
                    train_examples.append({
                        "qid": example["qid"],
                        "relT_pids": example["relT_pids"],
                        "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"],
                    })
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        example = ujson.loads(line)
                        train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len, label_mode)
    

    @classmethod
    def create_from_20relT_10neg_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, 
                                    label_mode, rank=-1, nranks=None):
        # canbe 20relT_10neg or 20T_10neg
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        assert label_mode in ["5", "10"]

        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))
        
        if rank == -1:
            train_examples = []
            with open(training_path, "r") as fin:
                for line in fin:
                    example = ujson.loads(line)
                    train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        example = ujson.loads(line)
                        train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
            
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len, label_mode)

    @classmethod
    def create_from_30relT_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, 
                                    label_mode, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        assert label_mode == "6"

        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))
        
        if rank == -1:
            train_examples = []
            with open(training_path, "r") as fin:
                for line in fin:
                    example = ujson.loads(line)
                    train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        example = ujson.loads(line)
                        train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
            
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len, label_mode)

    @classmethod
    def create_from_5relT_25neg_file(cls, queries_path, passages_path, training_path, tokenizer, max_query_len, max_passage_len, 
                                    label_mode, rank=-1, nranks=None):
        if rank != -1:
            assert rank in range(nranks) and nranks > 1
        assert label_mode in ["7", "8"]

        qid_to_query = {}
        with open(queries_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                qid, query = int(array[0]), array[1]
                qid_to_query[qid] = query 
        
        pid_to_passage = {}
        with open(passages_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if len(array) == 2:
                    pid, passage = int(array[0]), array[1]
                    pid_to_passage[pid] = passage 
                elif len(array) == 3:
                    pid, title, para = int(array[0]), array[1], array[2]
                    pid_to_passage[pid] = {"title": title, "para": para}
                else:
                    raise ValueError("array {}, with illegal length".format(array))

        if rank == -1:
            train_examples = []
            with open(training_path, "r") as fin:
                for line in fin:
                    new_example = {}
                    example = ujson.loads(line)
                    train_examples.append({
                        "qid": example["qid"],
                        "relT_pids": example["relT_pids"],
                        "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"],
                    })
        else:
            train_examples = []
            with open(training_path, "r") as fin:
                for line_idx, line in enumerate(fin):
                    if line_idx % nranks == rank:
                        example = ujson.loads(line)
                        train_examples.append({
                            "qid": example["qid"],
                            "relT_pids": example["relT_pids"],
                            "neg_pids": example["most_hard_pids"] + example["semi_hard_pids"]
                        })
        return cls(qid_to_query, pid_to_passage, train_examples, tokenizer, max_query_len, max_passage_len, label_mode)
    


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch

    queries_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv"
    passages_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/collection.tsv"
    #training_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/corase_to_fine_grained/relT_most_semi_hard.train.json" #"/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/argument_train/train_listwise.json"
    training_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/corase_to_fine_grained/20relT_10neg.train.json"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    #dataset = NwayDataset.create_from_relT_most_semi_hard_file(queries_path, passages_path, training_path, tokenizer, max_query_len=30, max_passage_len=128, label_mode="2")
    dataset = NwayDataset.create_from_20relT_10neg_file(queries_path, passages_path, training_path, tokenizer, max_query_len=30, max_passage_len=128, label_mode="5")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for b_idx, batch in enumerate(dataloader):
        for loop_idx, (q_token_ids, nway_token_ids) in enumerate(zip(batch["query"]["input_ids"], 
                                                            batch["nway_passages"]["input_ids"])):
            print("-----query-----")
            print("qid: ", batch["qid"][loop_idx])
            print(tokenizer.decode(q_token_ids, skip_special_tokens=True))
            print("nway_pids: \n", batch["nway_pids"][loop_idx])
            print("labels: \n", batch["labels"][loop_idx])
            for idx in range(len(nway_token_ids)):
                token_ids = nway_token_ids[idx]
                print("nway_passage: \n", tokenizer.decode(token_ids, skip_special_tokens=True))
                if idx == 3:
                    break
        if b_idx == 2:
            break