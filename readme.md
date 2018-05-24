# TGAN: A Tabular Data Synthesizer

TGAN is a tabular data synthesizer. It can generate fully synthetic data from real data. Currently, TGAN can generate numerical columns and categorical columns. This software can do random search for TGAN parameters  on multiple datasets using multiple GPUs. 

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

### Run Demo
```
> # Dowload data
> mkdir data
> wget -O data/census-train.csv https://s3.amazonaws.com/hdi-demos/tgan-demo/census-train.csv
> wget -O data/covertype-train.csv https://s3.amazonaws.com/hdi-demos/tgan-demo/covertype-train.csv
> python3 src/launcher.py demo_config.json
```

## How it works?

