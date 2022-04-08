# Curriculum Learning for Dense Retrieval Distillation (CL-DRD)
Hansi Zeng, Hamed Zamani, Vishwa, Vinay

This repo provides code, trained models for our SIGIR'22 paper: [Curriculum Learning for Dense Retrieval Distillation](https://hansizeng.github.io/). 
This paper introduced CL-DRD, a generic framework for optimizing dense retrieval models through knowledge distillation. Inspired by curriculum learning, CL-DRD follows an iterative optimization process in which the difficulty of knowledge distillation data produced by the teacher model increases at every iteration as shown in the following figure. We provided a simple implementation of this framework and achieved impressive results on MS MARCO-dev, TREC'19 and TREC'20 datasets. 

<p align="center">
  <img align="center" src="https://github.com/HansiZeng/CL-DRD/blob/main/CL-DRD%20figure.png" width="500" />
</p>
<p align="center">
  <b>Figure 1:</b> The data creation process in each iteration of curriculum learning based on knowledge distillation.
</p>

## Inference 
#### TASB + CL-DRD
First, Download the MS MARCO passage datasets from [MS MARCO Passage Ranking Dataset](https://microsoft.github.io/msmarco/Datasets). Second, download our pretrained model from [model](https://drive.google.com/file/d/1aC0-RSB6MU9v65OK1eh8yItyeG_nGQGU/view?usp=sharing). When data preparation steps finished, we can start to index collection:
``` 
python retriever/index_text.py --resume="path/to/pretrained/model" --passages_path="path/to/msmarco/collection.tsv" --index_dir="path/to/index_dir/"
```
It should take ~2.5h in a RTX8000 GPU. When indexing completed, you should see a new generated file named "checkpoint_120000.index" in your "path/to/index_dir/". Then we can start to retrieve top 1000 passages for eqch query from MS MARCO-dev set:
```
python retriever/retrieve_top_passages.py \
--queries_path="/path/to/queries.dev.tsv" \
--resume="path/to/pretrained/model" \
--index_path="path/to/index_dir/checkpoint_120000.index"
--output_path="path/to/passage_rankings.dev.run" \
```
Let's start evaluating the passage rankings in MS MARCO-dev set: 
```
python evaluation/retrieval_evaluator.py --ranking_path="path/to/passage_rankings.dev.run" --qrels_path="path/to/msmarco/qrels.dev.small.tsv"
```
You should get the output like this:
```
# msmarco-dev: 
{'MRR@10': 0.38174398508209395, 'QueriesWithRelevant@10': 4647, 'MRR@1000': 0.39230037918323496, 'QueriesWithRelevant@1000': 6848, 'Recall@50': 0.844663323782235, 'Recall@1000': 0.9788562559694365, 'nDCG@10': 0.44382220214795265, 'nDCG@100': 0.4963590703737573, 'MAP@1000': 0.38641636287519376, 'QueriesRanked': 6980}
```

