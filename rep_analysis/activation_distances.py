import h5py
import numpy as np
import pandas as pd
import sys
import sys
sys.path.append('../')
import argparse
import Globals
from tqdm import tqdm
from scipy.spatial import distance

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('activations_file',help='Path to activations file')
    parser.add_argument('-o','--output',default='activation_distances.parquet')
    args = parser.parse_args()
    distances = {'Model 1':[],'Model 2':[], 'Dataset':[],'Layer':[],'Euclidean Distance':[]}
    with h5py.File(args.activations_file,'r') as model_activations:
        for model1 in tqdm(model_activations.keys(),desc='Dataset'):
            for model2 in tqdm(model_activations.keys(),desc='Model2'):
                if model1 == model2:
                    continue
                else:
                    for dataset in model_activations[model1].keys():
                        model1_activations = Globals.get_model_responses(model1,args.activations_file,dataset)
                        model2_activations = Globals.get_model_responses(model2,args.activations_file,dataset)
                        for layer in Globals.LAYER_MAPPING.keys():
                            layer_activations1 = model1_activations[layer]
                            layer_activations2 = model2_activations[layer]
                            for i in range(len(layer_activations1)):
                                im_activation1 = layer_activations1[i]
                                im_activation2 = layer_activations2[i]
                                euclidean_distance = distance.euclidean(im_activation1.flatten(),im_activation2.flatten())
                                distances['Model 1'].append(model1)
                                distances['Model 2'].append(model2)
                                distances['Layer'].append(layer)
                                distances['Dataset'].append(dataset)
                                distances['Euclidean Distance'].append(euclidean_distance)
    
    distances_df = pd.DataFrame(distances)
    distances_df.to_parquet(args.output)
                            
                            
                    
