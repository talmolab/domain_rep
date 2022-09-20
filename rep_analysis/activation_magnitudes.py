import h5py
import numpy as np
import pandas as pd
import sys
import sys
sys.path.append('../')
import argparse
import Globals
import torch
import numpy as np
from tqdm import tqdm
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('activations_file',help='Path to activations file')
    parser.add_argument('-o','--output',default='activation_magnitudes.parquet')
    args = parser.parse_args()
    activation_magnitudes = {'Model':[], 'Dataset':[],'Layer':[],'Magnitude':[]}
    with h5py.File(args.activations_file,'r') as model_activations:
        for model in tqdm(model_activations.keys(),desc='Dataset'):
            for dataset in model_activations[model].keys():
                activations = Globals.get_model_responses(model,args.activations_file,dataset)
                for layer in activations.keys():
                    magnitudes = torch.norm(torch.Tensor(activations[layer]),p=2,dim=(-1)).numpy()
                    activation_magnitudes['Model'].extend([model for i in range(magnitudes.shape[0])])
                    activation_magnitudes['Dataset'].extend([dataset for i in range(magnitudes.shape[0])])
                    activation_magnitudes['Layer'].extend([layer for i in range(magnitudes.shape[0])])
                    activation_magnitudes['Magnitude'].extend(magnitudes)
    activation_magnitudes_df = pd.DataFrame(activation_magnitudes)
    activation_magnitudes_df.to_parquet(args.output)