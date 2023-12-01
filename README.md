# Contrastive Finetuning protein Language Models

This repo contains data and scripts to demonstrate how [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) can be used with protein Language Models, in particular [ESM](https://github.com/facebookresearch/esm/tree/main) models, as demonstrated in the paper "Optimizing protein language models with Sentence Transformers".

The two minimal examples are solubility and disorder predictions. Both scripts
* ```/scripts/solubility_search_seeds.py``` 
* ```/scripts/disorder_st_avg.py```

can be run in a fairly standard environment that would require the following packages
```
pip install -U transformers sentence-transformers accelerate bitsandbytes scikit-learn
```

Note that the scripts take the data from the ```/data``` folder and might require adjusting of the paths depending on the environment setting. 
Ideally the scripts should run on GPUs. For the ```disorder``` task in case of a large scale search one might consider caching the frozen residue level representations from ESM, 
as currently it automatically downloads those from huggingface on-the-fly.

