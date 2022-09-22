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
    parser.add_argument("-a","--analysis_files",nargs='+',action="store",choices=Globals.ANALYSIS_LINKS.keys(),default=Globals.ANALYSIS_LINKS.keys())
    parser.add_argument('-o','--output_dir',action="store",default='./analysis_files')
    args = parser.parse_args()
    if args.output_dir != '.':
        os.makedirs(args.output_dir,exist_ok=True)
    print(f'Downloading',', '.join(args.analysis_files))
    for analysis_file in args.analysis_files:
        file_name = gdown.download(Globals.ANALYSIS_LINKS[analysis_file],fuzzy=True)
        if file_name == None:
            raise ValueError(f"Could Not Download {analysis_file}")
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
        