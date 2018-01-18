import argparse
import data as dataset
from experiment_builder import run_experiment
from utils.argument_util import get_args

parser = argparse.ArgumentParser(description='Welcome to Face Transfer Via Autoencoders')


batch_size, num_gpus, continue_from_epoch, experiment_name, dropout_rate_value = get_args(parser=parser)
#set the data provider to use for the experiment
data = dataset.EmmaWatsonToDaisyRidleyDataset(batch_size=batch_size, num_gpus=num_gpus)
#init experiment
run_experiment(data, batch_size, num_gpus, continue_from_epoch, experiment_name, dropout_rate_value)