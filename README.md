# Contrastive Finetuning protein Language Models

This repo contains data and scripts to demonstrate how [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) can be used with protein Language Models, in particular [ESM](https://github.com/facebookresearch/esm/tree/main) models, as demonstrated in the paper "<i>Optimizing protein language models with Sentence Transformers</i>".

## Setup

Please note that this implementation requires GPUs.

```bash
git clone https://github.com/PeptoneLtd/contrastive-finetuning-plms.git
cd contrastive-finetuning-plms
pip install -r full_env.txt
```
## Usage
Two minimal examples showing how to train a solubility and disorder prediction are provided.
* ```scripts/solubility_search_seeds.py``` 
* ```scripts/disorder_st_avg.py```

Note that the scripts take the data from the ```data``` folder and might require adjusting of the paths depending on the environment setting. 
For the ```disorder``` task in case of a large scale search, one might consider caching the frozen residue level representations from ESM, 
as currently it automatically downloads those from [huggingface](https://huggingface.co/) on-the-fly.

## Licence
This source code is licensed under the Apache 2.0 license found in the ```LICENSE`` file in the root directory of this source tree.

