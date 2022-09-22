import logging
import torch
import numpy as np
import torchvision
import PIL
import warnings
import h5py
from tqdm import tqdm
from train_simclr.simclr_alexnet import SimCLRModel

SENSORIUM_DATASETS = ['21067-10-18', '22846-10-16', '23343-5-17', '23656-14-22', '23964-4-22', '26872-17-20']
VIS_LAYERS = ["VISp", "VISl", "VISal", "VISam", "VISpm", "VISrl"]
LAYER_MAPPING = {'conv1':2,'conv2':5,'conv3':8,'conv4':10,'conv5':12}
DOMAINS = ["Cremi","Quickdraw","Sketch","Infograph","Clipart","Painting","Real", "ImageNet"]
AIVC_IMAGES_PATH = 'datasets/stimulus_set.npy'
SENSORIUM_DATASET_PATH = "datasets/sensorium.hdf5"
DATASET_LINKS = {"Sensorium":"https://drive.google.com/file/d/1M1QQb9bpI1pswMcUwEqfSpuwrqHvmXGo/view?usp=sharing",
                 "AIVC-images":"https://drive.google.com/file/d/1MGsA5Eb7oZEM2JlaWj2eMEJhEfdpNzhO/view?usp=sharing",
                 "AIVC-calcium_raw":"https://drive.google.com/file/d/1LtZQa0zLuzZIYyNbJQQm-e6lPid2Fj2s/view?usp=sharing",
                 "AIVC-calcium_filtered": "https://drive.google.com/file/d/1LPkq__tzdZqYtSbBrAA9BoecqmIFaIcv/view?usp=sharing",
                 "AIVC-neuropixels_raw":"https://drive.google.com/file/d/1Ly0L6EEznTVdxK4UDrtK3m-OcFGILtsn/view?usp=sharing",
                 "AIVC-neuropixels_filtered":"https://drive.google.com/file/d/1LVSwbwrqy2Z1Q0N_HdnZvuTgaHbiEIbU/view?usp=sharing",
                 "all":"https://drive.google.com/drive/folders/1LMjrWxjZCJ2iOWLgr6_SKPMUJIYl8LjL?usp=sharing"
                }
CHECKPOINT_LINKS = {'ImageNet':"https://drive.google.com/file/d/1L8HJel7jVS1W8r-Ohbr2_v6XGThHftbz/view?usp=sharing",
                    'Real':'https://drive.google.com/file/d/1LI4UOwnwKMBpjMK3jUcQJffC7z9-kpEd/view?usp=sharing',
                    'Painting':'https://drive.google.com/file/d/1LBwskF-eM-iBDZm-h215gFtV5PccbfPL/view?usp=sharing',
                    'Clipart': "https://drive.google.com/file/d/1Kz0anzP72ZiJWCVR6Gwv6L6ZktkFvHmF/view?usp=sharing",
                    'Infograph': "https://drive.google.com/file/d/1LA1Tztq5dVHk972Kk-XQKKbKyBshmkQw/view?usp=sharing",
                    'Sketch': "https://drive.google.com/file/d/1LLDW6i0y55fFrdE4a1BVIVYrG0twaX6_/view?usp=sharing",
                    'Quickdraw': "https://drive.google.com/file/d/1LD5DyxOXEyPNaEu63EBrnRAvfqele8DW/view?usp=sharing",
                    'Cremi': "https://drive.google.com/file/d/1L6tugqshsQ3OXNnDgUlv612CVpkLJ6X6/view?usp=sharing",
                    'all': 'https://drive.google.com/drive/folders/1Krh9Qs74B40SfFLlLrmbmorQHw2dORrq?usp=sharing'
                   }
MODEL_ACTIVATION_LINKS = {'AIVC':"https://drive.google.com/file/d/1KopDi6aOeAaetbBr654HzLIhCLXPrIFf/view?usp=sharing",
                          'Sensorium':"https://drive.google.com/file/d/1Kq80LqGGZe3n1BqlnIw4pv5Oshfcl_90/view?usp=sharing",
                         }
ANALYSIS_LINKS = {"Activation_Distances-AIVC":"https://drive.google.com/file/d/1KLggo0-iNvxFtPZyU4R1NHMG5y3bVNWW/view?usp=sharing",
                  "Activation_Distances-Sensorium":"https://drive.google.com/file/d/1K9gn6JN5etvHu85BDwAgY1o5Cw-pNhKR/view?usp=sharing",
                  "Activation_Magnitudes-AIVC":"https://drive.google.com/file/d/1KniHoq5CkAtwNVHnAtnbIyzqcxs2RtUj/view?usp=sharing",
                  "Activation_Magnitudes-Sensorium":"https://drive.google.com/file/d/1KjnVoLY9x1FcTIeWoprTGeXJ-I-XGfZT/view?usp=sharing",
                  "Naturalness":"https://drive.google.com/file/d/1KhjlDB99FX5NZmAISUKHTLkd8h5Jnzi_/view?usp=sharing",
                  "Neural_Pred-Ephys":"https://drive.google.com/file/d/1Kf85kl_SP6lFFWmECCxmcAmxkonvz3M-/view?usp=sharing",
                  "Neural_Pred-Calcium":"https://drive.google.com/file/d/1KdZkpvodnlT3UPFwe1A7AInLStTQkz0r/view?usp=sharing",
                  "Neural_Pred-Sensorium":"https://drive.google.com/file/d/1KU632yBV9v-vTCnd7kS1MKvja4n4Pu1e/view?usp=sharing"
                 }
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def load_model(checkpoint_path, backbone_only = True):
    """Load a SimCLR contrastive model from a checkpoint.
    
    Args:
        checkpoint_path: Path to a saved .ckpt. If no path is given, activations are generated
        wrt a randomly initialized network.
        backbone_only: If True (the default), return only the backbone of the model.
            If False, return the full model, including projection heads.
    
    Returns:
        A pytorch model for the SimCLR-trained checkpoint.
    """
    if checkpoint_path=='Random' or checkpoint_path==None:
        model = SimCLRModel(None)
    else:
        model = SimCLRModel(None).load_from_checkpoint(checkpoint_path)
    if backbone_only:
        model = model.backbone[0]
    return model
def split_half(array_dict):
    
    #train1,train2,test1,test2 = train_test_split(np.transpose(train,(2,0,1)), np.transpose(test,(2,0,1)),test_size=0.5) #split by trials
    
    array_dict1 = {key:np.transpose(value,(2,0,1))[np.arange(0,50,2)] for key,value in array_dict.items()}
    array_dict1 = {key:np.mean(np.transpose(value,(1,2,0)),axis=-1) for key,value in array_dict1.items()}
    array_dict2 = {key:np.transpose(value,(2,0,1))[np.arange(1,50,2)] for key,value in array_dict.items()}
    array_dict2 = {key:np.mean(np.transpose(value,(1,2,0)),axis=-1) for key,value in array_dict2.items()}
    return array_dict1,array_dict2

def get_model_responses(model:str, activation_file:str, dataset:str):
    with h5py.File(activation_file,'r') as activations:
        model_activations = activations[model][dataset]
        model_activations = {key:np.array(activation) for key,activation in model_activations.items()}
    return  model_activations

def extract_neural_response(layer, mode=None, calcium_path=None,neuropixels_path = None, ims=np.arange(118)):
    if neuropixels_path != None:
        with h5py.File(neuropixels_path,'r') as neural_responses:
            VIS_responses = neural_responses[layer]
            per_specimen_response = {}
            for specimen in VIS_responses.keys():
                specimen_population = np.transpose(np.array(VIS_responses[specimen]),(1,0,2))#change to shape (num_ims,num_neurons,num_trials)
                per_specimen_response[specimen] = specimen_population[ims]
            return split_half(per_specimen_response)
    elif calcium_path != None:
        with h5py.File(hdf_path,'r') as neural_responses:
            layer_responses = neural_responses[layer]
            if mode=='average':
                averaged_responses = layer_responses['average']
                return {specimen:np.array(response)[ims] for specimen,response in averaged_responses.items()}
            elif mode=='even-odd':
                even_responses = layer_responses['even']
                odd_responses = layer_responses['odd']
                even_responses_dict = {specimen:np.array(response)[ims] for specimen,response in even_responses.items()}
                odd_responses_dict = {specimen:np.array(response)[ims] for specimen,response in odd_responses.items()}
                return even_responses_dict,odd_responses_dict
            else:
                raise ValueError("Mode must be one of `average` or `even-odd`")
    else:
        raise ValueError("Must provide either path to calcium neural data or ephys neural data")
def compile_sensorium_neural_responses(sensorium_neural_activations_path):
    with h5py.File(sensorium_neural_activations_path,'r') as sensorium_dataset:
        sensorium_responses = {specimen:{split:np.array(responses) 
                                         for split,responses in sensorium_dataset['Responses'][specimen].items()
                                        } 
                               for specimen in sensorium_dataset['Responses'].keys()
                              }
        sensorium_responses = {dataset:np.concatenate([sensorium_responses[dataset][split] for split in sensorium_responses[dataset].keys()]) for dataset in sensorium_responses.keys()}
    return sensorium_responses
    
def generate_model_activations(model, stimulus):
    """Compute model activations in response to a set of stimulus images.
    
    Args:
        model: A pytorch model (assumes AlexNet).
        stimulus: An iterable of images in torch.Tensors of shape (1, channels, width, height).
    
    Returns:
        A dictionary mapping convolutional layer names ("conv1", ..., "conv5") to activations
        as numpy arrays of shape (n_images, n_units).
        
        The number of images is the length of the input stimulus.
        
        The number of units varies per layer and is the flattened activation from that layer.
    """
    model_activations = {layer: None for layer in LAYER_MAPPING.keys()}
    for layer,ind in tqdm(LAYER_MAPPING.items(),desc='Conv Layer'):
        activation_model = model[:ind]
        activation_model.eval()
        activations = []
        for j in range(len(stimulus)):
            activation = activation_model(stimulus[j])
            activations.append(activation.detach().numpy().squeeze().flatten())
        activations = np.stack(activations)
        model_activations[layer] = activations
    return model_activations
def RunLengthWarning():
    print("WARNING: This script may take a while! We recommend you run it in the background using `tmux` or `nohup`.")
class NaturalScenes(torch.utils.data.Dataset):
    def __init__(self,file_path = f'datasets/stimulus_set.npy'):
        self.ims = np.load(file_path)
        self.size = 118
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize([256]),torchvision.transforms.CenterCrop(224),torchvision.transforms.Resize([64,64]),torchvision.transforms.ToTensor()])
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        im = PIL.Image.fromarray(self.ims[index])
        im = self.transform(im)
        im = im.unsqueeze(0)
        return im
def compile_sensorium_images(sensorium_dataset_path):
    with h5py.File(sensorium_dataset_path,'r') as sensorium_dataset_path:
        sensorium_images = sensorium_dataset['Images']
        sensorium_responses = sensorium_dataset['Responses']
        images_per_specimen = {}
        num_specimen = 1
        for specimen in sensorium_images.keys():
            print(f'{num_specimen}', sep = ' ', end = ' ')
            images = []
            for split in sensorium_images[specimen].keys():
                print(f'\n\t{split}', sep = ' ', end = ' ')
                if split =='final_test':
                    continue
                image_splits = np.array(sensorium_images[specimen][split])
                images.append(np.squeeze(image_splits,axis=1))
            images_per_specimen[specimen] = np.concatenate(images)
            print(f'\n{images_per_specimen[specimen].shape}')
            num_specimen+=1
        return images_per_specimen

class Sensorium(torch.utils.data.Dataset):
    def __init__(self,sensorium_dataset_path,ims, transform=True):
        self.ims = ims
        self.size = len(self.ims)
        if transform:
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.Resize([64,64]),torchvision.transforms.ToTensor()])
        else: 
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        im = self.ims[index]
        im = PIL.Image.fromarray(im)
        im = self.transforms(im)
        im = torch.cat([im,im,im],axis=0)
        im = im.unsqueeze(0)
        return im

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger