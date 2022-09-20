import os
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
from tqdm import tqdm
import sklearn
import h5py
import sys
sys.path.append('../')
import argparse
import Globals
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression

def spearman_brown(pearson_r):
    return (2*pearson_r)/(1+pearson_r)

def split_half(train,test):
    train1,train2,test1,test2 = train_test_split(train.T, test.T,test_size=0.5)
    
    train1 = train1.T
    # print(train1.shape)
    train2 = train2.T
    
    test1 = test1.T
    test2 = test2.T
    
    return (train1,train2), (test1,test2)

def inter_animal_consistency(model_activations_train, model_activations_test, specimen_response_train,specimen_response_test):
    # specimen_response_train_halves,specimen_response_test_halves = split_half(specimen_response_train,specimen_response_test)
    specimen_response_train1,specimen_response_train2 = specimen_response_train
    specimen_response_test1,specimen_response_test2 = specimen_response_test
    pls1 = PLSRegression(n_components=25)
    # print(model_activations_train.shape)
    pls1.fit(model_activations_train,specimen_response_train1)
    
    pls2 = PLSRegression(n_components=25)
    pls2.fit(model_activations_train,specimen_response_train2)
    
    predictions1 = pls1.predict(model_activations_test)
    predictions2 = pls2.predict(model_activations_test)
    
    r,p = sp.stats.pearsonr(predictions1.flatten(),specimen_response_test2.flatten())
    # print(r)
    mapping_r,mapping_p = sp.stats.pearsonr(predictions1.flatten(),predictions2.flatten())
    mapping_sbr = spearman_brown(mapping_r)
    # print(mapping_sbr)
    response_r,response_p = sp.stats.pearsonr(specimen_response_test1.flatten(),specimen_response_test2.flatten())
    response_sbr = spearman_brown(response_r)
    # print(response_sbr)
    denominator = np.sqrt(mapping_sbr * response_sbr)
    
    return r/denominator
    
def layer_predictivity(layer_activations,neural_activations):
    neural_response_train,neural_response_test = neural_activations
    layer_activations_train,layer_activations_test = layer_activations
    inter_animal_consistencies = {specimen:None for specimen in neural_response_train[0].keys()}
    for specimen in neural_response_train[0].keys():
        specimen_response_train = (responses[specimen] for responses in neural_response_train)
        specimen_response_test = (responses[specimen] for responses in neural_response_test)
        inter_animal_consistencies[specimen] = inter_animal_consistency(layer_activations_train, layer_activations_test, specimen_response_train, specimen_response_test)
    return np.median(list(inter_animal_consistencies.values()))

def model_predictivity(model_activations, neural_activations):
    # model_activations = Globals.extract_model_response(model, train_stim)
    # model_activations_test = extract_model_response(model, test_stim)
    layer_wise_predictivity = {layer:None for layer in model_activations.keys()}
    train_inds,test_inds = train_test_split(np.arange(118),test_size=0.5,train_size=0.5,shuffle=True)
    model_activations_train = {layer:activations[train_inds] for layer,activations in model_activations.items()}
    model_activations_test = {layer:activations[test_inds] for layer,activations in model_activations.items()}
    neural_activations_train = ({specimen:activation[train_inds] for specimen,activation in neural_activations[0].items()},
                                {specimen:activation[train_inds] for specimen,activation in neural_activations[1].items()}
                               )
    neural_activations_test = ({specimen:activation[test_inds] for specimen,activation in neural_activations[0].items()},
                               {specimen:activation[test_inds] for specimen,activation in neural_activations[1].items()}
                              )
    neural_activations = (neural_activations_train,neural_activations_test)
    for layer in model_activations_train.keys():
        layer_activations = (model_activations_train[layer],
                             model_activations_test[layer])
        layer_wise_predictivity[layer] = layer_predictivity(layer_activations,neural_activations)
    return layer_wise_predictivity    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--models',nargs='+',action='store',default=Globals.DOMAINS)
    parser.add_argument('-v','--vis_layers',nargs='+',action='store',default=Globals.VIS_LAYERS)
    parser.add_argument('-a','--activation_path',action='store',required=True)
    parser.add_argument('-c','--calcium_path',action='store')
    parser.add_argument('-n','--n_iters',action='store',default=100,type=int)
    parser.add_argument('-l','--log_path',action='store',default='neural_preds.log')
    parser.add_argument('-o','--output_path',action='store',default='neural_preds.parquet')
    args = parser.parse_args()
    Globals.RunLengthWarning()
    logger = Globals.setup_logger('neural_preds',args.log_path)
    results = {'Domain':[],"Vis_Layer":[],"Conv_layer":[],"Neural Predictivity":[]}
    for model in tqdm(args.models,desc='Models'):
        try:
            logger.info(f'Running {model} experiment for {args.n_iters} iterations') 
            model_activations = Globals.get_model_responses(model,args.activation_path,"AIVC")
            for vis_layer in args.vis_layers:
                neural_activations = Globals.extract_neural_response_calcium(vis_layer,args.calcium_path, "even_odd")
                for i in tqdm(range(args.n_iters),desc=f'{vis_layer}'):
                    model_predictivity = model_predictivity(model_activations,neural_activations)
                    for layer,pred in model_predictivity.items():
                        results['Domain'].append(model)
                        results['Vis_Layer'].append(vis_layer)
                        results['Conv_layer'].append(layer)
                        results['Neural Predictivity'].append(pred)
            logger.info(f'Experiment Complete.')
        except Exception as e:
            logger.error(e)
            continue
    results_df = pd.DataFrame(results)
    results_df.to_parquet(args.output_path)
                    
    