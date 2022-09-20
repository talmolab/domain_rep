import sys
sys.path.append('../')

import os
import torchvision
import PIL
import lightly
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import h5py
import logging
import Globals
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm

def save_model_activations(models_dict:dict,stimuli_dict,file_path,logger):
    with h5py.File(file_path,'a') as h5file:
        for key,model_path in tqdm(models_dict.items(),desc='models'):
            # try:
            logger.info(f'Generating {key} Activations...')
            model = Globals.load_model(model_path,backbone_only=True)
            model_group = h5file.require_group(key)
            model_group.attrs.create(name='Checkpoint Path', data=model_path)
            for stim_name,stimulus in tqdm(stimuli_dict.items(),desc='datasets'):
                logger.info(f'Gathering Activations in response to {stim_name}...')
                activations = Globals.generate_model_activations(model,stimulus)
                stim_group = model_group.require_group(stim_name)
                for layer,activation in activations.items():
                    logger.info(f'Saving activations to {stim_name}/{key}/{layer}...')
                    activation_dset = stim_group.require_dataset(name=layer,shape=activation.shape,dtype=activation.dtype,exact=False,data=activation)
            logger.info(f'{key} saved')
    return h5file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',choices=['Sensorium','AIVC'],help='Which imageset to generate activations in response to',required=True)
    parser.add_argument('-n', '--names',action='store',nargs='+',help='Names of models to be used as keys for hdf5 file')
    parser.add_argument('-c','--checkpoints',action='store',nargs='+',help='Checkpoint paths of models to generate activations. Keep in same order as `--name`')
    parser.add_argument('-l','--log_path',action='store',help='Path to text file for logging',default='model_activations.log')
    parser.add_argument('-f','--file_path',action='store',help='Path to model activations hdf5 file', default = './model_activations.hdf5')
    args = parser.parse_args()
    if args.dataset == 'Sensorium':
        images_per_specimen = compile_sensorium_images(f'../{Globals.SENSORIUM_DATASET_PATH}')
        dataset = {specimen:Globals.Sensorium(ims,transform=True) for specimen,ims in images_per_specimen.items()}
        Globals.RunLengthWarning()
    else:
        dataset={'AIVC':Globals.NaturalScenes(f'../{Globals.AIVC_IMAGES_PATH}')}
    logger = Globals.setup_logger('model_activations',args.log_path)
    models_dict = {model:ckpt for model,ckpt in zip(args.names,args.checkpoints)}
    save_model_activations(models_dict,dataset,args.file_path,logger)
        
    
            