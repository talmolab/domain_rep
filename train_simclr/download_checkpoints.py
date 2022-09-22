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
    parser.add_argument("-m","--models",nargs='+',action="store",choices=Globals.CHECKPOINT_LINKS.keys(),default=["all"])
    parser.add_argument('-o','--output_dir',action="store",default='./model_checkpoints')
    args = parser.parse_args()
    if args.output_dir != '.':
        os.makedirs(args.output_dir,exist_ok=True)
    if "all" in args.models:
        print('Downloading all checkpoints')
        checkpoints = gdown.download_folder(Globals.CHECKPOINT_LINKS['all'])
        print(checkpoints)
        for checkpoint in checkpoints:
            new_name = f'{args.output_dir}/{checkpoint.split("/")[-1]}'
            if args.output_dir != '.':
                print(f'Moving {checkpoint} to output directory to', new_name)
                os.rename(checkpoint,new_name)
            if new_name.endswith('.gz'):
                print(f'Uncompressing {new_name}')
                with gzip.open(new_name,'r') as f_in:        
                    with open(os.path.splitext(new_name)[0],'wb') as f_out:
                        shutil.copyfileobj(f_in,f_out)
                        print(f'{new_name} uncompressed to {f_out.name}')
                        os.remove(new_name)
        if args.output_dir != '.':
            print('Removing temp dir')
            os.rmdir(checkpoint.split('/')[-2])
        print('Download complete')
        exit()
    else:
        print(f'Downloading',', '.join(args.models))
        for model in args.models:
            file_name = gdown.download(Globals.CHECKPOINT_LINKS[model],fuzzy=True)
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
        print('Download complete')