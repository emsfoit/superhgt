# Universal HGT

**Important**: This Repo is a modified version of Heterogeneous Graph Transformer (HGT): [*github.com/acbull/pyHGT*](https://github.com/acbull/pyHGT), that is an experiment on Open Academic Graph (OAG) data.

## Overview

In this repository we extend the HGT work to build a unified system that handle serveral data domains. The script can be used to build HGT graph based on any provided data that could have the shape of a graph.

This is still an ongoing project...

## Potential Next Steps

- Connect benchmarking with attribution score (implement reverse engineering to understand the attention score by each node/edge)
- Capture the attention score across all edges
- Discover unseen edges between nodes
- Include sequencing

## Data Sources

We use the following data sources for our experiments on HGT:

- A sample of Open Academic Graph (OAG) data: [*AdaStruct > text > papers > OAG raw data - sample.zip*](https://www.dropbox.com/home/text/papers?preview=OAG+raw+data+-+sample.zip)

- [*Reddit data*](https://drive.google.com/file/d/14Zv3s-YPoAPhmQanWzUGJSK-AM_r61hm/view). The goal is to predict the subreddit of the post/comment, and/or predict what post the author is going to visit.

- Hashed customer journeys data from Google Analytics (GA) provided by Crealytics: [*AdaStruct > Marketing > Google Analytics – Proprietary Anonymized*](https://www.dropbox.com/home/marketing/Google%20Analytics%20%E2%80%93%20Proprietary%20Anonymized). The goal is to predict the next marketing channel touch point, and/or predict the likelihood of customer journeys that lead to basket conversions in online shopping.

## Setup

1. Fill in the `HGT_graph_params_{}.json` with the graph parameters, examples can be found in [models](../../../models)

2. Build the graph as in the example files, (this assumes that the nodes data is stored in csv files alike format with headers that are specified the .json file from the previous step):

```bash
python -m examples.build_HGT_graph_OAG
```

3. Train the model using the graph as in the examples:

```bash
python -m examples.train_HGT_model
```

**Note**: A new version that uses networkx is still in progress, use the following to run:

```bash
python -m examples.build_HGT_graph_OAGx
python -m examples.train_HGT_modelx
```

More info coming soon...





## Installation
Tested with python version 3.7.3


create a new envirmopent

``` python3 -m venv venv ```

then select it as the source of the project

``` source venv/bin/activate ```

Install the requirement.txt

``` pip install -r requirements.txt ```

Finally install Torch tools, if your device support cuda 11.3 run the following commands otherewise check pytorch [website](https://pytorch.org/get-started/locally/) and torch [geomeetrich](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)



``` pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html ``` 


```  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html ``` 




