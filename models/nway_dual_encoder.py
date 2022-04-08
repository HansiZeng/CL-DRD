
import torch
import torch.nn as nn 
from transformers import (AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup)

class NwayDualEncoder(nn.Module):
    def __init__(self, model_name_or_path, share_weights, in_batch_loss=False, all_in_batch_neg=True):
        super(NwayDualEncoder, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.share_weights = share_weights
        self.in_batch_loss = in_batch_loss
        self.all_in_batch_neg = all_in_batch_neg
        
        self.query_encoder = AutoModel.from_pretrained(model_name_or_path)

        if self.share_weights:
            self.passage_encoder = self.query_encoder
        else:
            self.passage_encoder = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, queries, nway_passages):
        """
        queries: each value shape: [bz, seq_len]
        nway_passage: each value shape: [bz, nway, seq_len]
        """
        query_reps = self.query_embs(queries) # [bz, D]
        nway_passage_reps = self.nway_passage_embs(nway_passages) #[bz, nway, D]
        
        
        if self.in_batch_loss:
            bz, nway, D = nway_passage_reps.shape
            neg_passage_idxs = [list(range(b_idx*nway)) + list(range((b_idx+1)*nway, bz*nway))
                                        for b_idx in range(bz)]
            if self.all_in_batch_neg:
                neg_passage_idxs = torch.LongTensor(neg_passage_idxs).to(nway_passage_reps.device)
            else:
                # hack
                neg_passage_idxs = torch.LongTensor(neg_passage_idxs).to(nway_passage_reps.device)
                xs = torch.arange(bz).view(-1,1).repeat(1,nway).to(nway_passage_reps.device)
                ys = torch.cat([torch.arange(0, (bz-1)*nway).view(bz-1,nway), torch.arange(0,nway).view(1,nway)],dim=0).to(nway_passage_reps.device)
                neg_passage_idxs = neg_passage_idxs[xs, ys] # [bz, nway]

            neg_passage_reps = nway_passage_reps.view(bz*nway,D)[neg_passage_idxs] #[bz, nway*(bz-1), D] or [bz, nway, D]
            nway_passage_reps = torch.cat([nway_passage_reps, neg_passage_reps], dim=1) # [bz, nway*bz, D] or [bz, 2*nway, D]
        
        assert query_reps.dim() == 2 and nway_passage_reps.dim() == 3
        logits = torch.sum(query_reps.unsqueeze(1) * nway_passage_reps, dim=-1) #[bz, nway] or [bz, nway*bz] or [bz, 2*nway]
        
        return logits

    def query_embs(self, queries):
        query_reps = self.query_encoder(**queries)[0][:, 0, :] 
        return query_reps

    def passage_embs(self, passages):
        passage_reps = self.passage_encoder(**passages)[0][:,0,:]
        return passage_reps

    def nway_passage_embs(self, nway_passages):
        input_ids, attention_mask = nway_passages["input_ids"], nway_passages["attention_mask"]
        bz, nway, seq_len = input_ids.shape 

        input_ids, attention_mask = input_ids.view(bz*nway, seq_len), attention_mask.view(bz*nway, seq_len)
        passage_reps = self.passage_encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        passage_reps = passage_reps.view(bz, nway, -1)

        return passage_reps
        
