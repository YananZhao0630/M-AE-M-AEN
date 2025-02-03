# Rethinking Graph Neural Networks from a Geometric Perspective of Node Features

This repository contains the code for our ICLR 2025 accepted paper, [_Rethinking Graph Neural Networks from a Geometric Perspective of Node Features_](#).

## Table of Contents
- [Requirements](#requirements)
- [Reproducing Results](#reproducing-results)
- [Reference](#reference)
- [Citation](#citation)

## Requirements
To install the required dependencies, refer to the `environment.yml` file.

## Reproducing Results
We take ACM-GCN as the basic example model. To reproduce the results in Table 2, run the following commands:


```bash
python ACM_GCN_heterphlic.py --dataset texas --graph_type new --prob_lambda 0.25 --train_epoch 40 --seed 10

python ACM_GCN_heterphlic.py --dataset wisconsin --graph_type new --prob_lambda 0.15 --train_epoch 20 --seed 2

python ACM_GCN_heterphlic.py --dataset squirrel --wd 5e-5 --graph_type new --prob_lambda 0.0001 --train_epoch 60 --seed 1

python ACM_GCN_heterphlic.py --dataset cornell --graph_type new --prob_lambda 0.01 --train_epoch 30 --seed 1

python ACM_GCN_heterphlic.py --dataset chameleon --wd 2e-5 --graph_type new --prob_lambda 0.08 --train_epoch 30 --seed 10

python ACM_GCN_heterphlic.py --dataset actor --wd 1e-4 --graph_type new --prob_lambda 0.1 --seed 10 --train_epoch 20
