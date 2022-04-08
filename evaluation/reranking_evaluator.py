from typing import List, Optional
import csv
import os 
import pickle
import sys 


import numpy as np 
import torch
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

#from utils import write_rankdata

def batch_to_cuda(batch):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda()
        if isinstance(batch[key],  BatchEncoding):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].cuda()
    return batch

class RerankingEvaluator():
    def __init__(self, qrel_path: str, mrr_at_k: List[int] = [10, 100], ndcg_at_k: List[int] = [10, 100], recall_at_k = [10, 100],
                save_to_csv: bool = True, show_progress_bar: bool = False, is_trec: bool = False):   
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
        
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.recall_at_k = recall_at_k
        if save_to_csv:
            self.csv_file = "evluation_result.csv"
            self.csv_header = ["epoch", "steps"] + ["mrr@{}".format(k) for k in  mrr_at_k] + ["ndcg@{}".format(k) for k in ndcg_at_k]
        
        self.show_progress_bar = show_progress_bar

    def __call__(self, model, dataloader, running_foloder: Optional[str] = None, epoch: int = -1, batch: int = -1):
        pass 

    def compute_metrics(self, model, dataloader, output_rerank_path=None, return_per_query=False, per_query_metrics_path=None, 
                        is_cross_encoder=False):   
        model.eval() 
        qid_to_rankdata = {}
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), disable= not self.show_progress_bar, desc="Reranking test", total=len(dataloader)):
                with torch.cuda.amp.autocast(enabled=True):
                    batch = batch_to_cuda(batch)

                    if is_cross_encoder:
                        scores = model(batch["query_passage"]).view(-1).cpu().tolist() 
                    else: 
                        query_features = model.query_embs(batch["query"]) # BxD
                        passage_features = model.passage_embs(batch["passage"]) 
                        scores = torch.sum(query_features*passage_features, dim=-1).cpu().tolist() 

                    qids = batch["qid"]
                    pids = batch["pid"]

                for (qid, pid, score) in zip(qids, pids, scores):
                    if qid not in qid_to_rankdata:
                        qid_to_rankdata[qid] = [(pid, score)]
                    else:
                        qid_to_rankdata[qid].append((pid, score))
                

        if output_rerank_path is not None:
            write_rankdata(qid_to_rankdata, output_rerank_path)

        qid_to_ranklist = {}
        for qid, rankdata in qid_to_rankdata.items():
            rankdata.sort(key=lambda x: x[1], reverse=True)
            ranklist = list(zip(*rankdata))[0]
            qid_to_ranklist[qid] = ranklist


        if return_per_query:
            local_dict, rr_per_candidate_depth,recall_per_candidate_depth,ndcg_per_candidate_depth, qidx_to_qid, qrels \
                                        = self._calculate_metrics_plain(qid_to_ranklist, self.qid_to_relevant_data, 
                                                                        return_per_query=return_per_query)
            if per_query_metrics_path is not None:
                self._output_per_query_metrics(qidx_to_qid, qrels, per_query_metrics_path, rr_per_candidate_depth, recall_per_candidate_depth, 
                                                ndcg_per_candidate_depth)

            return local_dict, (rr_per_candidate_depth, recall_per_candidate_depth, ndcg_per_candidate_depth)
        else:
            local_dict = self._calculate_metrics_plain(qid_to_ranklist, self.qid_to_relevant_data)
            return local_dict
    
    def direct_compute_metric(self, qid_to_ranklist, return_per_query=False, per_query_metrics_path=None):
        if return_per_query:
            local_dict, rr_per_candidate_depth,recall_per_candidate_depth,ndcg_per_candidate_depth, qidx_to_qid, qrels \
                                        = self._calculate_metrics_plain(qid_to_ranklist, self.qid_to_relevant_data, 
                                                                        return_per_query=return_per_query)
            if per_query_metrics_path is not None:
                self._output_per_query_metrics(qidx_to_qid, qrels, per_query_metrics_path, rr_per_candidate_depth, recall_per_candidate_depth, 
                                                ndcg_per_candidate_depth)

            return local_dict, (rr_per_candidate_depth, recall_per_candidate_depth, ndcg_per_candidate_depth)
        else:
            local_dict = self._calculate_metrics_plain(qid_to_ranklist, self.qid_to_relevant_data)
            return local_dict

    def _calculate_metrics_plain(self, ranking, qrels,binarization_point=1.0,return_per_query=False):
        '''
        calculate main evaluation metrics for the given results (without looking at candidates),
        returns a dict of metrics
        '''

        ranked_queries = len(ranking)

        qidx_to_qid = {idx:qid for idx, qid in enumerate(ranking)}
        
        rr_per_candidate_depth = np.zeros((len(self.recall_at_k),ranked_queries))
        rank_per_candidate_depth = np.zeros((len(self.mrr_at_k),ranked_queries))
        recall_per_candidate_depth = np.zeros((len(self.recall_at_k),ranked_queries))
        ndcg_per_candidate_depth = np.zeros((len(self.ndcg_at_k),ranked_queries))
        evaluated_queries = 0

        for query_index,(query_id,ranked_doc_ids) in enumerate(ranking.items()):
            if query_id in qrels:
                evaluated_queries += 1

                relevant_ids = np.array(list(qrels[query_id].keys())) # key, value guaranteed in same order
                relevant_grades = np.array(list(qrels[query_id].values()))
                sorted_relevant_grades = np.sort(relevant_grades)[::-1]

                num_relevant = relevant_ids.shape[0]
                np_rank = np.array(ranked_doc_ids)
                relevant_mask = np.in1d(np_rank,relevant_ids) # shape: (ranking_depth,) - type: bool

                binary_relevant = relevant_ids[relevant_grades >= binarization_point]
                binary_num_relevant = binary_relevant.shape[0]
                binary_relevant_mask = np.in1d(np_rank,binary_relevant) # shape: (ranking_depth,) - type: bool

                # check if we have a relevant document at all in the results -> if not skip and leave 0 
                if np.any(binary_relevant_mask):
                    
                    # now select the relevant ranks across the fixed ranks
                    ranks = np.arange(1,binary_relevant_mask.shape[0]+1)[binary_relevant_mask]


                    # mrr only the first relevant rank is used
                    first_rank = ranks[0]

                    for cut_indx, cutoff in enumerate(self.mrr_at_k):

                        curr_ranks = ranks.copy()
                        curr_ranks[curr_ranks > cutoff] = 0 

                        recall = (curr_ranks > 0).sum(axis=0) / binary_num_relevant
                        recall_per_candidate_depth[cut_indx,query_index] = recall

                        #
                        # mrr
                        #

                        # ignore ranks that are out of the interest area (leave 0)
                        if first_rank <= cutoff: 
                            rr_per_candidate_depth[cut_indx,query_index] = 1 / first_rank
                            rank_per_candidate_depth[cut_indx,query_index] = first_rank
                
                if np.any(relevant_mask):
                    
                    # now select the relevant ranks across the fixed ranks
                    ranks = np.arange(1,relevant_mask.shape[0]+1)[relevant_mask]

                    grades_per_rank = np.ndarray(ranks.shape[0],dtype=int)
                    for i,id in enumerate(np_rank[relevant_mask]):
                        grades_per_rank[i]=np.where(relevant_ids==id)[0]

                    grades_per_rank = relevant_grades[grades_per_rank]

                    #
                    # ndcg = dcg / idcg 
                    #
                    for cut_indx, cutoff in enumerate(self.ndcg_at_k):
                        #
                        # get idcg (from relevant_ids)
                        idcg = (sorted_relevant_grades[:cutoff] / np.log2(1 + np.arange(1,min(num_relevant,cutoff) + 1)))

                        curr_ranks = ranks.copy()
                        curr_ranks[curr_ranks > cutoff] = 0 

                        #coverage_per_candidate_depth[cut_indx, query_index] = (curr_ranks > 0).sum() / float(cutoff)

                        with np.errstate(divide='ignore', invalid='ignore'):
                            c = np.true_divide(grades_per_rank,np.log2(1 + curr_ranks))
                            c[c == np.inf] = 0
                            dcg = np.nan_to_num(c)

                        nDCG = dcg.sum(axis=-1) / idcg.sum()

                        ndcg_per_candidate_depth[cut_indx,query_index] = nDCG

        #avg_coverage = coverage_per_candidate_depth.sum(axis=-1) / evaluated_queries
        mrr = rr_per_candidate_depth.sum(axis=-1) / evaluated_queries
        relevant = (rr_per_candidate_depth > 0).sum(axis=-1)
        non_relevant = (rr_per_candidate_depth == 0).sum(axis=-1)

        """
        avg_rank=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), -1, rank_per_candidate_depth)
        avg_rank[np.isnan(avg_rank)]=0.

        median_rank=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), -1, rank_per_candidate_depth)
        median_rank[np.isnan(median_rank)]=0.
        """
        recall = recall_per_candidate_depth.sum(axis=-1) / evaluated_queries
        nDCG = ndcg_per_candidate_depth.sum(axis=-1) / evaluated_queries

        local_dict={}

        for cut_indx, cutoff in enumerate(self.mrr_at_k):

            local_dict['MRR@'+str(cutoff)] = mrr[cut_indx]
            local_dict['Recall@'+str(cutoff)] = recall[cut_indx]
            local_dict['QueriesWithNoRelevant@'+str(cutoff)] = non_relevant[cut_indx]
            local_dict['QueriesWithRelevant@'+str(cutoff)] = relevant[cut_indx]
            #local_dict['AverageRankGoldLabel@'+str(cutoff)] = avg_rank[cut_indx]
            #local_dict['MedianRankGoldLabel@'+str(cutoff)] = median_rank[cut_indx]
        
        for cut_indx, cutoff in enumerate(self.ndcg_at_k):
            #local_dict['Avg_coverage@'+str(cutoff)] = avg_coverage[cut_indx]
            local_dict['nDCG@'+str(cutoff)] = nDCG[cut_indx]

        local_dict['QueriesRanked'] = evaluated_queries
        
        if return_per_query:
            return local_dict,rr_per_candidate_depth,recall_per_candidate_depth,ndcg_per_candidate_depth, qidx_to_qid, qrels
        else:
            return local_dict

    def _output_per_query_metrics(self, qidx_to_qid, qrels, output_path,
                                    rr_per_candidate_depth, recall_per_candidate_depth, ndcg_per_candidate_depth):
        
        with open(output_path, "w") as fout:
            writer = csv.writer(fout)

            csv_header = ["query"] + \
                ["mrr@{}".format(k) for k in  self.mrr_at_k] + \
                ["recall@{}".format(k) for k in self.recall_at_k] + \
                ["ndcg@{}".format(k) for k in self.ndcg_at_k]
            writer.writerow(csv_header)

            for qidx in qidx_to_qid:
                qid = qidx_to_qid[qidx]
                if qid not in qrels:
                    continue

                csv_row = [qid]

                csv_row += ["{:.3f}".format(rr_per_candidate_depth[depth][qidx]) for depth in range(rr_per_candidate_depth.shape[0])]
                csv_row += ["{:.3f}".format(recall_per_candidate_depth[depth][qidx]) for depth in range(recall_per_candidate_depth.shape[0])]
                csv_row += ["{:.3f}".format(ndcg_per_candidate_depth[depth][qidx]) for depth in range(ndcg_per_candidate_depth.shape[0])]

                writer.writerow(csv_row)

        

        

if __name__ == "__main__":
    import sys 
    sys.path += ["../"]
    from models.dual_encoder import DualEncoder
    from models.cross_encoder import CrossEncoder
    from dataset import MsMacroRerankingDataset, RerankingDataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModel
    #from utils import MetricMonitor

    #model_name_or_path = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    cross_encoder = True
    if cross_encoder:
        model_name_or_path = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        model = CrossEncoder(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        model_name_or_path = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        tokenizer_name_or_path = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        model = DualEncoder(model_name_or_path, share_weights=True)
        
    qrels_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/sbert/qrels.dev.tsv"  #/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/trec-19/2019qrels-pass.txt"
    ranking_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/sbert/qidpids.dev.tsv" #"/home/hzeng/ir-research/my-msmacro-passage/evaluation/runs/trec-19/tas256-trec19-top1000.run"
    queries_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv" #"/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/trec-19/msmarco-test2019-queries.tsv"
    passages_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/collection.tsv"

    model.cuda() 

    #dataset = MsMacroRerankingDataset(reranking_path, tokenizer, is_cross_encoder=True, max_len=300)
    dataset = RerankingDataset(ranking_path, queries_path, passages_path, tokenizer, is_cross_encoder=cross_encoder, max_len=300)
    dataloader = DataLoader(dataset, batch_size=512, collate_fn=dataset.collate_fn, num_workers=4)

    evaluator = RerankingEvaluator(qrels_path, is_trec=False, show_progress_bar=True)
    #local_dict, _  = evaluator.compute_metrics(model, dataloader, output_rerank_path="runs/trec-19/tas256-ce-rerank-top1000.run", return_per_query=True, 
                                            #per_query_metrics_path="runs/trec-19/tas256-ce-rerank-top1000_per_q_metrics.csv", is_cross_encoder=True)
    #print(local_dict)
    local_dict = evaluator.compute_metrics(model, dataloader, is_cross_encoder=cross_encoder)
    print(local_dict)
            