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

import torch 
import torch.nn as nn 
from transformers.tokenization_utils_base import BatchEncoding
import torch.distributed as dist
from transformers import (AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup)
import numpy as np 
from tqdm import tqdm, trange


from dataset import NwayDataset, RerankingDataset
from utils import MetricMonitor, AverageMeter, is_first_worker
from models.nway_dual_encoder import NwayDualEncoder
from losses import lambda_mrr_loss

import torch.cuda.amp as amp 

#EXPERIMENT_FOLDER = "/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/"
BLOCK_SIZE = 50_000
HIDDEN_SIZE = 768
PAD_REL_PID = -1
target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#if not os.path.exists(EXPERIMENT_FOLDER):
#os.mkdir(EXPERIMENT_FOLDER)

def set_env(args):
    args.distributed = False
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group("nccl")
        args.device = device
        args.distributed = dist.get_world_size() > 1
        args.nranks = dist.get_world_size()
        args.ngpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cuda")

def save_model(model, output_dir, save_name, args, optimizer=None):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.bin"))

def save_checkpoint(state, is_best, filename="checkpoint.pt.tar"):
    torch.save(state, filename)
    if is_best:
        p = Path(filename)
        shutil.copyfile(filename, os.path.join(p.parent, "model_best.pth.tar"))

def batch_to_device(batch, target_device: torch.device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
        if isinstance(batch[key], dict) or isinstance(batch[key], BatchEncoding):
            for sub_key in batch[key]:
                if isinstance(batch[key][sub_key], torch.Tensor):
                    batch[key][sub_key] = batch[key][sub_key].to(target_device)

    return batch

def write_train_logs(epoch, step, loss_val, mrr_val, recall_val, lr, filename, cutoff=10):
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write(f"epoch\tstep\tloss_val\tmrr@{cutoff}\trecall@{cutoff}\tlr\n")
    else:
        with open(filename, "a") as f:
            f.write(f"{epoch}\t{step}\t{loss_val:.3f}\t{mrr_val:.3f}\t{recall_val:.3f}\t{lr:.10f}\n")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/queries.train.tsv")
    parser.add_argument("--collection_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/collection.tsv")
    parser.add_argument("--training_path", default="/mnt/nfs/scratch1/hzeng/datasets/msmarco-passage/corase_to_fine_grained/10relT_20neg.train.json")
    parser.add_argument("--experiment_folder", default="/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/")

    parser.add_argument("--model_name_or_path", default="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
    parser.add_argument("--tokenizer_name_or_path", default="distilbert-base-uncased")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--model_checkpoint", default="/mnt/nfs/scratch1/hzeng/my-msmarco-passage/experiments/multistep-curriculum/experiment_04-06_191527/models/checkpoint_250000.pth.tar")
    parser.add_argument("--seed", default=4680, type=bool)

    parser.add_argument("--show_progress", default=True, type=bool)
    parser.add_argument("--run_folder", default="experiment")
    parser.add_argument("--log_dir", default="log/")
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--evaluate_steps", default=10000, type=int)
    parser.add_argument("--model_save_dir", default="models")
    
    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--warmup_steps", default=4000, type=int)

    parser.add_argument("--query_max_len", default=30, type=int)
    parser.add_argument("--passage_max_len", default=256, type=int)
    parser.add_argument("--use_fp16", default=True, type=bool)
    parser.add_argument("--train_batch_size", default=8, type=int)

    parser.add_argument("--share_weights", default=False)
    parser.add_argument("--label_mode", default="9", type=str)
    parser.add_argument("--in_batch_loss", action="store_true", default=False)
    parser.add_argument("--all_in_batch_neg",  action="store_true", default=False)


    parser.add_argument("--n_gpu", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()

    if args.local_rank < 1:
        if not os.path.exists(args.run_folder):
            os.mkdir(args.run_folder)

        args.run_folder = os.path.join(args.experiment_folder, args.run_folder+"_"+time.strftime("%m-%d_%H%M%S", time.localtime()))
        if not os.path.exists(args.run_folder):
            os.mkdir(args.run_folder)
        args.log_dir = os.path.join(args.run_folder, args.log_dir)
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)
        
        args.model_save_dir = os.path.join(args.run_folder, args.model_save_dir)
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)

        config_path = os.path.join(args.run_folder, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(args.__dict__, f)

        fh = logging.FileHandler(filename=os.path.join(args.run_folder, "train_logs.log"))
        logger.addHandler(fh)

        assert args.model_checkpoint != None

    return args

def train(args):
    mrr_avg_meter = AverageMeter()
    recall_avg_meter = AverageMeter()
    loss_avg_meter = AverageMeter()
    
    # dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if args.distributed:
        assert args.train_batch_size % args.nranks == 0
        assert args.label_mode in ["3", "5", "6", "8", "9", "10"]
        if args.label_mode in ["3", "9"]:
            train_dataset = NwayDataset.create_from_10relT_20neg_file(args.queries_path, args.collection_path, args.training_path, tokenizer, 
                                                    max_query_len=args.query_max_len, max_passage_len=args.passage_max_len,
                                                    label_mode=args.label_mode, rank=args.local_rank, nranks=args.nranks)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size//args.nranks, shuffle=True, 
                                                    num_workers=1,collate_fn=train_dataset.collate_fn, drop_last=True)
        elif args.label_mode in ["5", "10"]:
            train_dataset = NwayDataset.create_from_20relT_10neg_file(args.queries_path, args.collection_path, args.training_path, tokenizer, 
                                                    max_query_len=args.query_max_len, max_passage_len=args.passage_max_len,
                                                    label_mode=args.label_mode, rank=args.local_rank, nranks=args.nranks)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size//args.nranks, shuffle=True, 
                                                    num_workers=1,collate_fn=train_dataset.collate_fn, drop_last=True)
        elif args.label_mode == "6":
            train_dataset = NwayDataset.create_from_30relT_file(args.queries_path, args.collection_path, args.training_path, tokenizer, 
                                                    max_query_len=args.query_max_len, max_passage_len=args.passage_max_len,
                                                    label_mode=args.label_mode, rank=args.local_rank, nranks=args.nranks)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size//args.nranks, shuffle=True, 
                                                    num_workers=1,collate_fn=train_dataset.collate_fn, drop_last=True)
        elif args.label_mode in ["7", "8"]:
            train_dataset = NwayDataset.create_from_5relT_25neg_file(args.queries_path, args.collection_path, args.training_path, tokenizer, 
                                                max_query_len=args.query_max_len, max_passage_len=args.passage_max_len,
                                                label_mode=args.label_mode, rank=args.local_rank, nranks=args.nranks)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size//args.nranks, shuffle=True, 
                                                    num_workers=1,collate_fn=train_dataset.collate_fn, drop_last=True)
        else:
            raise ValueError(f"label mode {args.label_mode} not implemented")
        # not implement dev dataset
    else:
        assert args.label_mode in ["3", "5", "6"]
        if args.label_mode == "3":
            train_dataset = NwayDataset.create_from_10relT_20neg_file(args.queries_path, args.collection_path, args.training_path, tokenizer, 
                                                    max_query_len=args.query_max_len, max_passage_len=args.passage_max_len,
                                                    label_mode="3")
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, 
                                                    num_workers=1,collate_fn=train_dataset.collate_fn, drop_last=True)
        elif args.label_mode == "5":
            train_dataset = NwayDataset.create_from_20relT_10neg_file(args.queries_path, args.collection_path, args.training_path, tokenizer, 
                                                    max_query_len=args.query_max_len, max_passage_len=args.passage_max_len,
                                                    label_mode="5")
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, 
                                                    num_workers=1,collate_fn=train_dataset.collate_fn, drop_last=True)
        elif args.label_mode == "6":
            train_dataset = NwayDataset.create_from_30relT_file(args.queries_path, args.collection_path, args.training_path, tokenizer, 
                                                    max_query_len=args.query_max_len, max_passage_len=args.passage_max_len,
                                                    label_mode="6", rank=args.local_rank, nranks=args.nranks)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, 
                                                    num_workers=1,collate_fn=train_dataset.collate_fn, drop_last=True)
        else:
            raise ValueError(f"label mode {args.label_mode} not implemented")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4,
                                                        collate_fn=train_dataset.collate_fn)

    # model
    model = NwayDualEncoder(args.model_name_or_path, share_weights=args.share_weights, in_batch_loss=args.in_batch_loss,
                            all_in_batch_neg=args.all_in_batch_neg)
    model.to(target_device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.local_rank],
                                                            output_device=args.local_rank)
    if args.distributed:
        dist.barrier()

    # optim
    t_total = len(train_dataloader) * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    test_monitor = MetricMonitor()
    # start training 
    if args.local_rank < 1:
        logger.info("***** Start training *****")
        logger.info(f"Total number of samples = {len(train_dataset)}")
        logger.info(f"Total training steps = {t_total}, num epochs = {args.num_train_epochs}, lr = {args.learning_rate}")
        logger.info("use fp16 = {}, share_weights = {}, label_mode = {}, in_batch_loss = {}, all_in_batch_neg = {}".format(
            args.use_fp16, args.share_weights, args.label_mode, args.in_batch_loss, args.all_in_batch_neg))
        if args.distributed:
            logger.info(f"process rank: {args.local_rank}, ngpu: {args.ngpu}, world size: {args.nranks}")
        logger.info("training path = {}, model_name_or_path = {}".format(args.training_path, args.model_name_or_path))
        if args.model_checkpoint:
            logger.info("load pretrained model from: {}".format(args.model_checkpoint))

    set_seed(args)

    mrr, recall = 0, 0
    checked_topk = 10
    model.zero_grad()
    global_step = 0
    best_metric = 0. # record mrr@10
    start_epoch = 0

    # resume training
    if args.resume:
        assert args.model_checkpoint == None 
        if os.path.isfile(args.resume):
            if args.distributed:
                loc = "cuda:{}".format(args.local_rank)
                checkpoint = torch.load(args.resume, map_location=loc)
            else:
                checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["state_dict"]) # only load model state_dict
            global_step = checkpoint["global_step"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] - 1
            logger.info("resume checkpoint from ==> {}".format(args.resume))
            # free gpu memory
            del checkpoint
        else:
            print("not checkpoint found at ===> {}".format(args.resume))
    
    # load model checkpoint
    if args.model_checkpoint:
        assert args.resume == None
        if os.path.isfile(args.model_checkpoint):
            if args.distributed:
                loc = "cuda:{}".format(args.local_rank)
                checkpoint = torch.load(args.model_checkpoint, map_location=loc)
                model.load_state_dict(checkpoint["state_dict"])
            else:
                checkpoint = torch.load(args.model_checkpoint)
                print("load parallel wrapped model.")
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            # free gpu memory
            del checkpoint
        else:
            raise ValueError("not model checkpoint foundt at ====> {}".format(args.model_checkpoint))


    # start training 
    if args.use_fp16:
        scaler = amp.GradScaler()
    for epoch_idx, epoch in enumerate(trange(start_epoch, args.num_train_epochs, desc="Epoch")):
        for step, batch in enumerate(tqdm(train_dataloader, desc="query_epoch", disable=args.local_rank >=1, total=len(train_dataloader))):
            model.train()
            batch = batch_to_device(batch, target_device)
            with amp.autocast(enabled=args.use_fp16):
                pred_logits = model(batch["query"], batch["nway_passages"])
                if args.in_batch_loss:
                    bz, all_nway = pred_logits.shape 
                    assert all_nway % bz == 0 
                    if args.all_in_batch_neg:
                        nway = all_nway // bz
                        gt_labels = torch.cat([batch["labels"], -0.5*torch.ones(bz, (bz-1)*nway).to(target_device)], dim=-1)
                    else:
                        nway = all_nway // 2
                        gt_labels = torch.cat([batch["labels"], -0.5*torch.ones(bz, nway).to(target_device)], dim=-1)
                    loss = lambda_mrr_loss(pred_logits, gt_labels) # -1 is the mask_indicator for lambda_mrr_loss()
                else:
                    loss = lambda_mrr_loss(pred_logits, batch["labels"])
            if args.use_fp16:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if args.local_rank < 1:
                # train loss & mrr & recall 
                if args.in_batch_loss:
                    labels = gt_labels.cpu()
                else:
                    labels = batch["labels"].cpu()
                with torch.no_grad():
                    logits = pred_logits.cpu()
                _, sorted_idxs = logits.sort(descending=True, dim=-1)
                labels = torch.gather(labels, dim=-1, index=sorted_idxs).numpy()
                b_first_pos = np.where(labels==1)[1]
                remain_first_pos = b_first_pos[b_first_pos<checked_topk]
                if len(remain_first_pos) == 0:
                    b_mrr, b_recall = 0., 0.
                else:
                    b_mrr = np.sum(1 / (remain_first_pos + 1.)) / len(b_first_pos)
                    b_recall = len(remain_first_pos) / len(b_first_pos)
                mrr_avg_meter.update(b_mrr)
                recall_avg_meter.update(b_recall)
                train_loss = loss.item() 
                loss_avg_meter.update(train_loss)

            global_step += 1 
            
            if args.local_rank < 1:
                if global_step % args.logging_steps == 0:
                    cur_loss = loss_avg_meter.avg 
                    cur_mrr = mrr_avg_meter.avg 
                    cur_recall = recall_avg_meter.avg
                    write_train_logs(epoch+1, global_step, cur_loss, cur_mrr, cur_recall, scheduler.get_lr()[0], 
                                    filename=os.path.join(args.log_dir, "train_logs.log"), cutoff=checked_topk)
                    
                    loss_avg_meter.reset()
                    mrr_avg_meter.reset() 
                    recall_avg_meter.reset()

                if (global_step) % args.evaluate_steps == 0:
                    save_checkpoint({
                                'epoch': epoch+1,
                                'global_step': global_step,
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                            }, is_best=False, filename=os.path.join(args.model_save_dir, f'checkpoint_{global_step}.pth.tar'))



if __name__ == "__main__":
    args = get_args()
    set_env(args)
    train(args)