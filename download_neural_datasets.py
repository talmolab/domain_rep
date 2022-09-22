import gdown
import argparse
import os
import sys
import gzip
import shutil
import Globals
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--datasets",nargs='+',action="store",choices=Globals.DATASET_LINKS.keys(),default=["all"])
    parser.add_argument('-o','--output_dir',action="store",default='./neural_data')
    args = parser.parse_args()
    if args.output_dir != '.':
        os.makedirs(args.output_dir,exist_ok=True)
    if "all" in args.datasets:
        print('Downloading all neural datasets')
        datasets = gdown.download_folder(Globals.DATASET_LINKS['all'])
        print(datasets)
        for dataset in datasets:
            new_name = f'{args.output_dir}/{dataset.split("/")[-1]}'
            if args.output_dir != '.':
                print(f'Moving {dataset} to output directory to', new_name)
                os.rename(dataset,new_name)
            if new_name.endswith('.gz'):
                print(f'Uncompressing {new_name}')
                with gzip.open(new_name,'r') as f_in:        
                    with open(os.path.splitext(new_name)[0],'wb') as f_out:
                        shutil.copyfileobj(f_in,f_out)
                        print(f'{new_name} uncompressed to {f_out.name}')
                        os.remove(new_name)
        if args.output_dir != '.':
            print('Removing temp dir')
            os.rmdir(dataset.split('/')[-2])
        print('Download complete')
        exit()
    else:
        print(f'Downloading',', '.join(args.datasets))
        for dataset in args.datasets:
            file_name = gdown.download(Globals.DATASET_LINKS[dataset],fuzzy=True)
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
        