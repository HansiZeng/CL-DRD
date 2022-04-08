import os 
import sys 
from pathlib import Path 
from typing import Union, List, Dict

from torch.utils.data import DataLoader, Dataset 

from .utils import load_passages, load_queries


class RerankingDataset(Dataset):
    """
    ranking_path format: qid<\t>pid<\t>rank<\t>score<\n> or qid<\t>pid<\t>score<\n> or qid<\t>pid<\n>
    """
    def __init__(self, ranking_path, queries_path, passages_path, tokenizer, is_cross_encoder, query_first=True, **kwargs):
        self.qid_pid_pairs = [] 
        with open(ranking_path, "r") as f:
            for line in f:
                array = line.strip().split("\t")
                if query_first:
                    qid, pid = int(array[0]), int(array[1])
                    self.qid_pid_pairs.append((qid, pid))
                else:
                    pid, qid = int(array[0]), int(array[1])
                    self.qid_pid_pairs.append((qid, pid))
        
        self.qid_to_query = load_queries(queries_path)
        self.pid_to_passage = load_passages(passages_path)

        self.is_cross_encoder = is_cross_encoder
        if is_cross_encoder:
            self.seq_max_len = kwargs["max_len"]
        else:
            self.query_max_len = kwargs["query_max_len"]
            self.passage_max_len = kwargs["passage_max_len"]
        
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        qid, pid = self.qid_pid_pairs[idx]
        query = self.qid_to_query[qid]
        passage = self.pid_to_passage[pid]

        if isinstance(passage, str):
            pass 
        elif isinstance(passage, dict):
            passage = passage["title"] + " " + self.tokenizer.sep_token + " " + passage["para"]
        else:
            raise ValueError("passage {} donot have desired format.".format(passage))
        return {
            "qid": qid,
            "pid": pid,
            "query": query,
            "passage": passage 
        }

    def __len__(self):
        return len(self.qid_pid_pairs)
        
    def collate_fn(self, batch):
        qids, pids, queries, passages = [], [], [], []
        for elem in batch:
            qids.append(elem["qid"])
            pids.append(elem["pid"])
            queries.append(elem["query"])
            passages.append(elem["passage"])

        if self.is_cross_encoder:
            query_passages = self.tokenizer(queries, passages, padding=True, truncation='longest_first', 
                                            return_tensors="pt", max_length=self.seq_max_len)
            return {
                "qid": qids,
                "pid": pids,
                "query_passage": query_passages
            }
        else:
            queries = self.tokenizer(queries, padding=True, truncation='longest_first', 
                                return_tensors="pt", max_length=self.query_max_len)
            passages = self.tokenizer(passages,  padding=True, truncation='longest_first', 
                                return_tensors="pt", max_length=self.passage_max_len)

            return {
                "qid": qids, 
                "pid": pids,
                "query": queries,
                "passage": passages
            }