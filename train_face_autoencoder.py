import argparse
from pathlib import Path

import data_providers as dataset
from experiment_builder import run_experiment
from utils.argument_util import get_args

parser = argparse.ArgumentParser(description='Welcome to Face Transfer Via Autoencoders')


batch_size, num_gpus, continue_from_epoch, experiment_name, dropout_rate_value = get_args(parser=parser)
#set the data provider to use for the experiment
training_data_dir = Path("data/training_data")

data = dataset.AToBDataset(batch_size=batch_size, num_gpus=num_gpus,
                           path_images_A=training_data_dir / Path("ryan_gosling"),
                           path_images_B=training_data_dir / Path("daisy_ridley"))
#init experiment
run_experiment(data, batch_size, num_gpus, continue_from_epoch, experiment_name, dropout_rate_value)