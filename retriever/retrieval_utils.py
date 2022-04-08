import os 
import pickle 
import glob 
import pathlib 
from typing import Union, List, Dict
import copy
import logging
logger = logging.getLogger(__name__)
import sys 
sys.path += ["./"]

from tqdm import tqdm, trange
import torch 
import faiss 
import numpy as np 
from timeit import default_timer as timer

from utils import batch_to_cuda

def batch_to_device(batch, target_device: torch.device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
        if isinstance(batch[key], dict):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].to(target_device)
    return batch

def get_embeddings_from_scratch(model, dataloader, use_fp16, is_query, show_progress_bar):
    embeddings = []
    embeddings_ids = []
    model.eval()
    for _, batch in tqdm(enumerate(dataloader), disable= not show_progress_bar, 
                                desc=f"encode # {len(dataloader)} seqs"):
        #print(batch["seq"]["input_ids"][:5])
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_fp16):
                batch = batch_to_cuda(batch)
                if is_query:
                    reps = model.query_embs(batch["seq"])
                else:
                    reps = model.passage_embs(batch["seq"])
            
                text_ids = batch["id"]

        embeddings.append(reps.cpu().numpy())
        assert isinstance(text_ids, list)
        embeddings_ids.extend(text_ids)
        
    embeddings = np.concatenate(embeddings)

    assert len(embeddings_ids) == embeddings.shape[0]
    assert isinstance(embeddings_ids[0], int)
    print(f"# nan in embeddings: {np.sum(np.isnan(embeddings))}")
    #assert embeddings.dtype == np.float16 if use_fp16 else np.float32  [Warning] ebedding.dtype != np.float16 even I enable the autocast. Strange !!!

    return embeddings, embeddings_ids

def write_embeddings_to_memmap(embeddings, embeddings_ids, block_size, run_folder, text_type, use_fp16):
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
    print(f"write {text_type} embeddings to {run_folder}")
    hidden_size = embeddings.shape[1]
    dtype = embeddings.dtype
    
    cur_idx = 0
    stored_block = 0
    per_block_last_idx = 0
    blocks_last_idx = []
    ebd_idxs = []
    reps_array = np.memmap(os.path.join(run_folder, "{}_reps_{}.npy".format(text_type, stored_block+1)), dtype=np.float16 if use_fp16 else dtype,
                            mode="w+", shape=(block_size, hidden_size))
    for sample_idx in tqdm(range(embeddings.shape[0]), disable=False, desc=f"write {text_type} embedding", total=embeddings.shape[0]):
        ebd_idxs.append(sample_idx)
        if sample_idx >= (stored_block+1) * block_size:
            blocks_last_idx.append(per_block_last_idx)
            per_block_last_idx = 0
            stored_block += 1

            reps_array = np.memmap(os.path.join(run_folder, "{}_reps_{}.npy".format(text_type, stored_block+1)), dtype=np.float16 if use_fp16 else dtype,
                            mode="w+", shape=(block_size, hidden_size))
            reps_array[sample_idx - (stored_block+1) * block_size] = embeddings[sample_idx]
        else:
            reps_array[sample_idx - (stored_block+1) * block_size] = embeddings[sample_idx]    
        per_block_last_idx += 1
    blocks_last_idx.append(per_block_last_idx)

    print("final per_block_last_idx: {}".format(per_block_last_idx))
    meta = {"embedding_ids": embeddings_ids, "ebd_idxs": ebd_idxs, "ntotal": sample_idx+1,
            "stored_block": stored_block+1, "blocks_last_idx": blocks_last_idx}
    print({"ntotal": sample_idx+1, "stored_block": stored_block+1, "blocks_last_idx": blocks_last_idx})
    with open(os.path.join(run_folder, f"meta_{text_type}.pkl"), "wb") as f:
        pickle.dump(meta, f)
    
def read_embeddings_from_memmap(run_folder, text_type, block_size, hidden_size, use_fp16) -> (np.ndarray, List[int], Dict[int, str]):
    # please check ebd_idxs is continuous from 0 to num_of_emb_example 
    # please check embedding_ids is np.int64
    with open(os.path.join(run_folder, f"meta_{text_type}.pkl"), "rb") as f:
        meta = pickle.load(f)
    embedding_ids = meta["embedding_ids"]
    ebd_idxs = meta["ebd_idxs"]
    blocks_last_idx = meta["blocks_last_idx"]
    #ebdidx_to_id = {idx:ebid for idx, ebid in zip(ebd_idxs, embedding_ids)}
    ebdid_to_idx = {id:idx for id, idx in zip(embedding_ids, ebd_idxs)}
    storage = []
    for fid in trange(1,len(glob.glob(os.path.join(run_folder,f"{text_type}_reps_*")))+1, desc="read embeddings from memmap"):
        storage.append(np.memmap(os.path.join(run_folder, f"{text_type}_reps_{fid}.npy"),
                        dtype=np.float16 if use_fp16 else np.float32, mode="r", shape=(block_size, hidden_size))[:blocks_last_idx[fid-1], :])
    embeddings = np.concatenate(storage)
    print(f"read {text_type} embeddings which have {embeddings.shape[0]} examples from folder {run_folder}")
    assert embeddings.shape[0] == len(ebdid_to_idx)

    return embeddings, embedding_ids, ebdid_to_idx

def construct_flatindex_from_embeddings(embeddings, ids):
    hidden_size = embeddings.shape[1]
    print('embedding shape: ' + str(embeddings.shape))
    index = faiss.index_factory(hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)
    if ids is not None:
        if isinstance(ids, list):
            ids = np.array(ids)
        ids = ids.astype(np.int64)
        print(ids.shape, ids.dtype)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)
    return index

def index_retrieve(index, query_embeddings, topk, batch=None):
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        nn_scores, nearest_neighbors = index.search(query_embeddings, topk)
    else:
        query_offset_base = 0
        pbar = tqdm(total=len(query_embeddings))
        nearest_neighbors = []
        nn_scores = []
        while query_offset_base < len(query_embeddings):
            batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
            batch_nn_scores, batch_nn = index.search(batch_query_embeddings, topk)
            nearest_neighbors.extend(batch_nn.tolist())
            nn_scores.extend(batch_nn_scores.tolist())
            query_offset_base += len(batch_query_embeddings)
            pbar.update(len(batch_query_embeddings))
        pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nn_scores, nearest_neighbors

def convert_index_to_gpu(index, faiss_gpu_index, useFloat16=False):
    if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
        faiss_gpu_index = faiss_gpu_index[0]
    if isinstance(faiss_gpu_index, int):
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = useFloat16
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
    else:
        global gpu_resources
        if len(gpu_resources) == 0:
            import torch
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256*1024*1024)
                gpu_resources.append(res)

        assert isinstance(faiss_gpu_index, list)
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = useFloat16
        for i in faiss_gpu_index:
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index