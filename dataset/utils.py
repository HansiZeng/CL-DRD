import os 


def load_queries(path):
    qid_to_query = {}
    with open(path, "r") as f:
        for line in f:
            qid, query = line.strip().split("\t")
            qid_to_query[int(qid)] = query
    
    return qid_to_query

def load_passages(path):
    pid_to_passage = {}
    with open(path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        array = line.strip().split("\t")
        if len(array) == 2:
            pid, passage = int(array[0]), array[1]
            pid_to_passage[pid] = passage 
        elif len(array) == 3:
            pid, title, para = int(array[0]), array[1], array[2]
            pid_to_passage[pid] = {"title": title, "para": para}
        else:
            raise ValueError("array {}, with illegal length".format(array))

    return pid_to_passage
