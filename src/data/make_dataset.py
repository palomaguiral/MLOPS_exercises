# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import numpy as np
import torch

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):

    #input: data/raw
    #output: data/processed
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    content = [ ]
    for i in range(5):
        content.append(np.load(input_filepath + f"/train_{i}.npz", allow_pickle=True))
        data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
        targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
    train_output_path=output_filepath +  "/train_data.npz"
    np.savez(train_output_path,images=data, labels=targets)

    
    content = np.load(input_filepath + "/test.npz", allow_pickle=True)
    data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
    data = torch.nn.functional.normalize(data)
    targets = torch.tensor(content['labels'])
    test_output_path=output_filepath +  "/test.npz"
    np.savez(test_output_path,images=data, labels=targets)
    

if __name__ == '__main__':
    input_filepath = "C:/Users/usuario/MLOps/nuevo_repo/data/raw"
    output_filepath = "C:/Users/usuario/MLOps/nuevo_repo/data/processed"

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main(input_filepath, output_filepath)
