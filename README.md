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
