import os 
import sys 
from pathlib import Path 
from typing import Union, List, Dict

from torch.utils.data import DataLoader, Dataset 

class SequenceDataset(Dataset):
    def __init__(self, id_to_seq: Dict[int, str], tokenizer, max_length: int, is_query: bool):
        super(SequenceDataset, self).__init__()
        self.tokenizer = tokenizer 
        self.max_length = max_length
        self.is_query = is_query 

        self.id_seq_pair = [(sid, seq) for sid, seq in id_to_seq.items()]

    def __getitem__(self, idx):
        sid, seq = self.id_seq_pair[idx]
        if not self.is_query: # it should be a doc 
            pass

        return {
            "seq": seq,
            "id": sid 
        }
    
    def __len__(self):
        return len(self.id_seq_pair)

    @classmethod
    def create_from_seqs_file(cls, seqs_file, tokenizer, max_length, is_query):
        id_to_seq = {}
        with open(seqs_file, "r") as f:
            for line in f:
                if is_query:
                    sid, seq = line.strip().split("\t")
                else:
                    sid, seq = line.strip().split("\t")

                id_to_seq[int(sid)] = seq

        return cls(id_to_seq, tokenizer, max_length, is_query)

    def collate_fn(self, batch):
        ids, seqs = [], []
        for elem in batch:
            seqs.append(elem["seq"])
            ids.append(elem["id"])
        
        seqs = self.tokenizer(seqs, padding=True, truncation='longest_first', 
                            return_tensors="pt", max_length=self.max_length)
        return {
            "seq": seqs,
            "id": ids
        }