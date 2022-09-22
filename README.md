# This is the codebase to reproduce the results in the paper "Natural Image Statistics and Modeling Neural Representations"

## Reproducing the results:
You have the option of either reproducing each step yourself or there are download scripts provided at each step for downloading our results.
There are n steps to reproducing. At each step, cd into the corresponding directories - `train_simclr`, `generate_activations` and `rep_analysis`:

1. Install the conda env provided using `conda env create -f neural_pred.yml`. Then activate the environment using `conda activate domain_rep`

2. Download the neural datasets needed for the pipeline using `python download_neural_datasets.py -d [DATASETS] -o [OUTPUT_DIR]` See python download_neural_datasets -h` for more info.

3. Train the simclr model on whatever datasets you want using `python train.py [DATASET_PATH] --options`. Currently all the flags are set to the hyperparameters we used but if you'd like to change them, see `python train_simclr/train.py -h`.

4. Generate model activations to each dataset using `python generate_model_activations.py -d [dataset] -m [model names] -c [checkpoint paths]. See `python generate_model_activations.py`

5. Once you've generated the model activations, you can use any of the scripts located in `rep_analysis` to reproduce our analysis.
