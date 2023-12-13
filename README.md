# Contrastive Finetuning protein Language Models

This repo contains data and scripts to demonstrate how [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) can be used with protein Language Models, in particular [ESM](https://github.com/facebookresearch/esm/tree/main) models, as demonstrated in the paper <i>[Optimizing protein language models with Sentence Transformers](https://www.mlsb.io/papers_2023/Optimizing_protein_language_models_with_Sentence_Transformers.pdf)</i>, NeurIPS (2023).

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

## Citations <a name="citations"></a>

If you use this work in your research, please cite the the relevant software:

```BibTeX
@software{Redl_Contrastive_Finetuning_protein,
author = {Redl, Istvan Redl and Airoldi, Fabio Airoldi and Bottaro, Sandro Bottaro and Chung, Albert Chung and Dutton, Oliver Dutton and Fisicaro, Carlo Fisicaro and Henderson, Louie Henderson and Hofmann, Falk Hoffmann and Invernizzi, Michele Invernizzi and Ruschetta, Stefano Ruschetta},
title = {{Contrastive Finetuning protein Language Models}},
url = {https://github.com/PeptoneLtd/contrastive-finetuning-plms},
version = {0.2.0}
}
```

## Licence
This source code is licensed under the Apache 2.0 license found in the ```LICENSE``` file in the root directory of this source tree.

