# TGAN: A Tabular Data Synthesizer

TGAN is a tabular data synthesizer. It can generate fully synthetic data from real data. Currently, TGAN can generate numerical columns and categorical columns. This software can do random search for TGAN parameters  on multiple datasets using multiple GPUs. 

## Citation

If you use TGAN, please cite the following work:

> Lei Xu, Kalyan Veeramachaneni. 2018. Synthesizing Tabular Data using Generative Adversarial Networks.

```
@article{xu2018synthesizing,
  title={Synthesizing Tabular Data using Generative Adversarial Networks},
  author={Xu, Lei and Veeramachaneni, Kalyan},
  journal={arXiv preprint arXiv:1811.11264},
  year={2018}
}
```


## Quick Start
### requirements

- pandas
- numpy
- sklearn
- tensorflow-gpu
- tensorpack

```
> pip3 install pandas numpy sklearn tensorflow-gpu tensorpack
```

### Run Demo
This demo shows how to generate synthetic version of census and covertype dataset. Generated synthetic datasets will be stored in `expdir/census` and `expdir/covertype`, while the GAN models will be stored in `train_log`. 

```
> # Dowload data
> mkdir data
> wget -O data/census-train.csv https://s3.amazonaws.com/hdi-demos/tgan-demo/census-train.csv
> wget -O data/covertype-train.csv https://s3.amazonaws.com/hdi-demos/tgan-demo/covertype-train.csv
> python3 src/launcher.py demo_config.json
```

This demo runs around 20 hours on our server which has 2 GTX 1080 GPUs. 

## How it works?
### Datasets
The input to this software is a csv file and a json config. 

- csv file should not have header or index. It should only contain numerical columns and categorical columns.  It should not contain any missing value. 
- json file specifies a list of experiments. Each experiment includes
	- **name**: the name of an experiment. We will create a folder in this name. 
	- **num\_random\_search**: iterations of random hyper parameter search. 
	- **train\_csv**: path to the training csv file. 
	- **continuous\_cols**: a list of column indexes which is numerical. (Index starts from 0.)
	- **epoch**: Number of epoches to train the model.
	- **steps\_per\_epoch**: Number of optimization steps in each epoch. 
	- **output\_epoch**: How many models to evaluate? output\_epoch <= min(epoch, 5). 
	- **sample\_rows**: In evaluation, how many rows should the synthesizer generate? 

Example JSON

```
[{
    'name': 'census',
    'num_random_search': 10,
    'train_csv': 'data/census-train.csv',
    'continuous_cols': [0, 5, 16, 17, 18, 29, 38],
    'epoch': 5,
    'steps_per_epoch': 10000,
    'output_epoch': 3,
    'sample_rows': 10000
}, ...]
```

### Training and Automated Evaluation

#### Split Training data. 
We split training data into two parts. `expdir/{name}/data_I.csv` has 80% data while `expdir/{name}/data_II.csv` has 20% data. We use data\_I to train GAN and use data\_II to evaluate the model. 

#### Random Search
All tunable hyper parameters are listed in `src/TGAN_synthesize.py:23 tunable_variables`. In each iteration of random search, we randomly select a value for each tunable variable. We than train the model and generate synthetic data using the last output_epoch stored models. 

#### Evaluation
We train a decision tree classifier (with max depth=20) on the synthetic dataset and compute the accuracy of that model on data\_II. User can pick the best hyper parameter for a dataset by reading `expdir/{name}/exp-result.csv`. 

### Outputs

#### expdir/{name}
- **data\_I.csv, data\_II.csv**: splited data. 
- **exp-params.json**: hyper parameters selected in random search.
- **exp-result.csv**: has num\_random\_search rows and output\_epoch columns, showing the classification accuracy of default classifier trained on a synthetic data and tested on data\_II. 
- **train.npz**: convert data\_I.csv to an npz file.
- **synthetic{iter}_{epoch}.csv/npz**: Synthetic data. iter is the random search iteration id. epoch is the training epoch. 

#### train\_log/TGAN\_synthsizer:{name}-{iter}
- This folder contains training log and last 5 models. 

## TODOs
- Select evaluation metrics for different experiment, e.g. F1, accuracy, AUC, etc. 
- Select default classifier for evaluation. 
- Support regression. 
