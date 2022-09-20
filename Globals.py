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
DOMAINS = ["Cremi", "Quickdraw","Sketch","Infograph","Clipart","Painting","Real", "ImageNet"]
AIVC_IMAGES_PATH = 'datasets/stimulus_set.npy'
SENSORIUM_DATASET_PATH = "datasets/sensorium.hdf5"
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

def get_model_responses(model:str, activation_file:str, dataset:str):
    with h5py.File(activation_file,'r') as activations:
        model_activations = activations[model][dataset]
        model_activations = {key:np.array(activation) for key,activation in model_activations.items()}
    return  model_activations

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
    warnings.warn("This script may take a while! We recommend you run it in the background using `tmux` or `nohup`.",ResourceWarning)
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