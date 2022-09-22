import gdown
import argparse
import os
import sys
import gzip
import shutil
sys.path.append('../')
import Globals
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--datasets",nargs='+',action="store",choices=Globals.MODEL_ACTIVATION_LINKS.keys(),default=Globals.MODEL_ACTIVATION_LINKS.keys())
    parser.add_argument('-o','--output_dir',action="store",default='./model_activations')
    args = parser.parse_args()
    if args.output_dir != '.':
        os.makedirs(args.output_dir,exist_ok=True)
    print(f'Downloading',', '.join(args.datasets))
    for dataset in args.datasets:
        file_name = gdown.download(Globals.MODEL_ACTIVATION_LINKS[dataset],fuzzy=True)
        if file_name == None:
            raise ValueError(f"Could Not Download {dataset}")
        if file_name.endswith('.gz'):
            print(f'Uncompressing {file_name}')
            with gzip.open(file_name,'r') as f_in:        
                with open(os.path.splitext(file_name)[0],'wb') as f_out:
                    shutil.copyfileobj(f_in,f_out)
                    print(f'{file_name} uncompressed to {f_out.name}')
                    os.remove(file_name)
            file_name = os.path.splitext(file_name)[0]
        if args.output_dir != ".":
            os.rename(file_name,f'{args.output_dir}/{file_name}')
        