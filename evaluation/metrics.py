from tqdm import tqdm 

def recall(qid_to_ranked_pids, qid_to_relevant_pids, cutoff=None, verbose=True):
    assert len(qid_to_relevant_pids) <= len(qid_to_ranked_pids)
    
    recall = 0.
    for qid in qid_to_ranked_pids:
        if qid not in qid_to_relevant_pids:
            continue
        else:
            relevant_pids = set(qid_to_relevant_pids[qid])
            ranked_pids = set(qid_to_ranked_pids[qid][:cutoff]) if cutoff is not None else set(qid_to_ranked_pids[qid])

            recall += len(relevant_pids & ranked_pids) / len(relevant_pids) 
    
    recall /= len(qid_to_relevant_pids)

    if verbose is True:
        print(f"# query with judgement = {len(qid_to_relevant_pids)}, # query be ranked = {len(qid_to_ranked_pids)}")
    
    return recall 

def recall_from_file(ranklist_path, qrels_path, cutoff=None, verbose=True):
    qid_to_ranked_pids = {}
    with open(ranklist_path, "r") as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="read ranklist", disable= not verbose):
        array = line.strip().split("\t")
        if len(array) == 2:
            qid, pid = int(array[0]), int(array[1])
        elif len(array) == 3:
            qid, pid, score = int(array[0]), int(array[1]), float(array[2])
        elif len(array) == 4:
            qid, pid, rank, score = int(array[0]), int(array[1]), int(array[2]) float(array[3])
        else:
            raise ValueError("length of array should be 2,3 but get {len(array)}")
        
        if qid not in qid_to_ranked_pids:
            qid_to_ranked_pids[qid] = [pid]
        else:
            qid_to_ranked_pids[qid] += [pid]
        
    qid_to_relevant_pids = {}
    with open(qrels_path, "r") as f:
        for line in f:
            qid, _, pid, _ = line.strip().split("\t")
            qid, pid = int(qid), int(pid)

            if qid not in qid_to_relevant_pids:
                qid_to_relevant_pids[qid] = [pid] 
            else:
                qid_to_relevant_pids[qid] += [pid]
    
    return recall(qid_to_ranked_pids=qid_to_ranked_pids, qid_to_relevant_pids=qid_to_relevant_pids, cutoff=cutoff, verbose=verbose)
    



if __name__ == "__main__":
    qrels_path = "/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/qrels.train.tsv"
    ranklist_path = "./train_query_ranklist_0.tsv"

    print(recall_from_file(ranklist_path, qrels_path, verbose=True))