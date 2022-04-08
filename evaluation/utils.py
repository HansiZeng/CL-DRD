import sys 
#sys.path += ["./"]
from typing import Union, List
import os 
from pathlib import Path
from timeit import default_timer as timer

from tqdm import tqdm, trange
import torch 
from torch import Tensor, device
import numpy as np
import faiss


def batch_to_device(batch, target_device: device):
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
        if isinstance(batch[key], dict):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], Tensor):
                    batch[key][sub_key] = batch[key][sub_key].to(target_device)
    return batch

def test_from_file(ranklist_path, qrel_path,  mrr_at_k: List[int] = [10, 100], ndcg_at_k: List[int] = [10, 100]):
    print("start test ranking performance")
    qid_to_ranked_pids = {}
    print(f"read ranklist data from {ranklist_path}")
    with open(ranklist_path, "r") as f:
        for line in f:
            array = line.strip().split("\t")
            if len(array) == 2:
                qid, pid = array[0], array[1]
            elif len(array) == 3:
                qid, pid, _ = array[0], array[1]
            elif len(array) == 4:
                qid, pid, _, _ = array[0], array[1]
            else:
                raise ValueError("not legal array length.")
            
            if qid not in qid_to_ranked_pids:
                qid_to_ranked_pids[qid] = [pid]
            else:
                qid_to_ranked_pids[qid].append(pid)
    
    print(f"read relevant data from {qrel_path}")
    qid_to_relevant_data = {}
    with open(qrel_path, "r") as f:
        for line in f:
            qid, _, pid, grade = line.strip().split("\t")
            if qid not in qid_to_relevant_data:
                qid_to_relevant_data[qid] = {}
                qid_to_relevant_data[qid][pid] = float(grade)
            else:
                qid_to_relevant_data[qid][pid] = float(grade)

    cutoff_to_mrr, cutoff_to_ndcg = test(qid_to_ranked_pids, qid_to_relevant_data, mrr_at_k, ndcg_at_k)

    return cutoff_to_mrr, cutoff_to_ndcg


def test(qid_to_ranklist, qid_to_relevant_data, mrr_at_k: List[int] = [10, 100], ndcg_at_k: List[int] = [10, 100]):
    reference_qids = set(qid_to_relevant_data.keys())
    candidate_qids = set(qid_to_ranklist.keys())
    if not reference_qids.issubset(candidate_qids):
        raise ValueError("Reference qids is not the subset of candidate qids")
    cutoff_to_queries_mrr = {k: np.zeros(len(qid_to_relevant_data.keys())) for k in mrr_at_k}
    cutoff_to_queries_ndcg = {k: np.zeros(len(qid_to_relevant_data.keys())) for k in ndcg_at_k}

    for query_idx, qid in enumerate(qid_to_relevant_data.keys()):
        ranklist = qid_to_ranklist[qid]
        relevant_pids = list(qid_to_relevant_data[qid].keys())
        relevant_grades = list(qid_to_relevant_data[qid].values())
        sorted_relevant_grades = np.sort(relevant_grades)[::-1]

        # mrr 
        relevant_masks = np.in1d(ranklist, relevant_pids)
        relevant_positions = np.arange(1, len(relevant_masks)+1)[relevant_masks]
        if len(relevant_positions) == 0: # if there is not relevant passages, ignore
            continue
        for cutoff in mrr_at_k:
            if min(relevant_positions) <= cutoff:
                cutoff_to_queries_mrr[cutoff][query_idx] = 1. / min(relevant_positions)

        # ndcg
        num_relevant = len(relevant_pids)
        relevant_grades_curlist = np.zeros(len(ranklist))
        for idx, rel_pid in enumerate(relevant_pids):
            if rel_pid in ranklist:
                pid_pos = ranklist.index(rel_pid)
                relevant_grades_curlist[pid_pos] = relevant_grades[idx]
            
        for cutoff in ndcg_at_k:
            idcg = np.sum(sorted_relevant_grades[:min(num_relevant, cutoff)] / np.log2(1 + np.arange(1, min(num_relevant, cutoff)+1)))
            cur_grades = relevant_grades_curlist[:cutoff]
            dcg = np.sum(cur_grades / np.log2(1 + np.arange(1, len(cur_grades)+1)))
            ndcg = dcg / idcg 
            
            cutoff_to_queries_ndcg[cutoff][query_idx] = ndcg 

    cutoff_to_mrr = {cutoff: np.mean(mrrs) for cutoff, mrrs in cutoff_to_queries_mrr.items()}
    cutoff_to_ndcg = {cutoff: np.mean(ndcgs) for cutoff, ndcgs in  cutoff_to_queries_ndcg.items()}

    return cutoff_to_mrr, cutoff_to_ndcg   

def index_retrieve(index, query_embeddings, topk, batch=None):
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        _, nearest_neighbors = index.search(query_embeddings, topk)
    else:
        query_offset_base = 0
        pbar = tqdm(total=len(query_embeddings))
        nearest_neighbors = []
        while query_offset_base < len(query_embeddings):
            batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
            batch_nn = index.search(batch_query_embeddings, topk)[1]
            nearest_neighbors.extend(batch_nn.tolist())
            query_offset_base += len(batch_query_embeddings)
            pbar.update(len(batch_query_embeddings))
        pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nearest_neighbors

def generate_qid_to_ranklist(query_embeddings: np.ndarray, query_ids: Union[np.ndarray, List[int]], index, 
                                epoch, cutoff=100, batch=32, output_path=None):
    queries_nearest_neighbors = index_retrieve(index, query_embeddings, cutoff, batch=batch)
    qid_to_ranked_pids = {}
    for qid, passage_ids in zip(query_ids, queries_nearest_neighbors):
        assert qid not in qid_to_ranked_pids
        qid_to_ranked_pids[qid] = passage_ids

    if output_path != None:
        fn = output_path[:-4] + f"_{epoch}" + output_path[-4:]
        with open(fn, "w") as f:
            for qid in qid_to_ranked_pids:
                for pid in qid_to_ranked_pids[qid]:
                    f.write(f"{qid}\t{pid}\n")

    return qid_to_ranked_pids, fn

def write_rankdata(qid_to_rankdata, output_path):
    """
    {qid: {(pid1, score1), (pid2, score2), ...} 
    """
    p = Path(output_path)
    if not os.path.exists(p.parent):
        os.mkdir(p.parent)
        
    with open(output_path, "w") as fout:
        for qid in qid_to_rankdata:
            rankdata = qid_to_rankdata[qid]
            rankdata = sorted(rankdata, key=lambda x: x[1], reverse=True)

            for i, (pid, score) in enumerate(rankdata):
                fout.write(f"{qid}\t{pid}\t{i+1}\t{score}\n")


    self.qid_to_relevant_data = {}
    with open(qrel_path, "r") as f:
        for line in f:
            if is_trec:
                qid, _, pid, grade = line.strip().split(" ")
            else:
                qid, _, pid, grade = line.strip().split("\t")
            
            qid, pid = int(qid), int(pid)
            if float(grade) <= 0.00001:
                continue

            if qid not in self.qid_to_relevant_data:
                self.qid_to_relevant_data[qid] = {}
                self.qid_to_relevant_data[qid][pid] = float(grade)
            else:
                self.qid_to_relevant_data[qid][pid] = float(grade)
