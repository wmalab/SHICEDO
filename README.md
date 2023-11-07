# SHICEDO: Single-cell Hi-C Resolution Enhancement with Reduced Over-smoothing
In this work, we introduce SHICEDO, a novel deep-learning model specifically designed for enhancing scHi-C resolution while addressing the over-smoothing issue. Built on a generative adversarial network (GAN) framework, SHICEDO's generator can process low-resolution scHi-C input of varying scales and sizes, generating an enhanced scHi-C matrix as the output. Leveraging our prior work on bulk Hi-C data, EnHiC, we have incorporated and improved its rank-one feature extraction and reconstruction techniques, along with our new feature refinement modules, into the SHICEDO framework.

![Model_Overview](figure/Model_figure.png)
> *This is a citation from a [source](https://www.example.com).*
- [SHICEDO](#SHICEDO:-Single-cell-Hi-C-Resolution-Enhancement-with-Reduced-Over-smoothing)
  - [Installation](#Installation)
  - [Download processed data](#Download-processed-data)
  - [Training](#Training)
  - [Prediction](#Prediction)
  - [Prediction with pre-trained model](#Prediction-with-pre-trained-model)
  - [Evaluation](#Evaluation)
  - [Data preprocessing](#Data-preprocessing)
  - [Heatmap and loss visualization](#Heatmap-and-loss-visualization)
- [Demo](#Demo)
## Installation
For the environment: Install PyTorch based on the CUDA version of your machine. <br>
Please check [PyTorch](https://pytorch.org/get-started/previous-versions/) for details<br>
<br>
In this demo, the machine has CUDA Version: 11.6<br>
To create SCHICEDO environment, use: <br>
`conda env create -f SCHICEDO_environment.yml` <br>
To activate this environment, use<br>
`conda activate SCHICEDO`
## Download processed data
The processed data is available at the following link:<br>
[Download processed data](https://drive.google.com/drive/folders/1EgkzPoNG-s_pi3SKOFG_YFslpIar_Bht?usp=sharing).<br>
There are two processed data available, in the following example, we will demo with processed Lee et al. dataset in folder Lee <br>
The downloaded data may be compressed in different files, please move the files into one folder after Extract <br>
1. `mkdir data`<br>
2. Please download the processed data to the data folder and use the correct path in the script for data loading.<br>
If you wish to preprocess other datasets. Please check the data preprocessing section

## Data preprocessing
If you wish to process raw data, please run the following command:<br> 
In this example, we show how to process the Nagano et al raw data is available at [Download raw data](https://drive.google.com/drive/folders/1UihcMw9DNR35Wps6FKVw-5EbiR7Tw55u?usp=sharing).<br>
`cd data_preprocessing`<br>
` ./data_preprocessing.sh` <br>
data_preprocessing.sh will run 6 scripts to save processed data: <br>
1. Filter the cells based on contact number `python data_filter.py`<br>
2. Filter out the inter-chromosomal interactions `python filter_true_data.py`<br>
3. Downsampling the matrix to generate low-resolution input `python down_sampling_sciHiC.py`<br>
4. Run Rscrip to do Bandnorm `Rscript bandnorm.R`<br>
5. Organize normalized result `python run_bandnorm.py`<br>
6. Divide large matrixes into submatrices and save as torch tensor `python generate_input.py`<br>

## Training
For optimal performance when training on new data, parameter fine-tuning is essential.<br>
The model and date setting were the same as described in the paper.  <br>
After choosing suitable hyper-parameters, the model can be trained with the following command: <br>
`python test_train.py` <br>

## Prediction
After training, Enhanced scHi-C can predict with the following command:<br>
`python test_prediction.py` <br>
Users can also use the provided pre-trained model to make predictions. <br>
Please change the corresponding model loading path in the test_prediction.py file.<br>

## Prediction with pre-trained model
Users can use the provided pre-trained model to make the prediction:<br>
1. `mkdir pretrained_model`<br>
2. Please download the pretrained model to the pretrained_model folder and use the correct path in the script [Download pre-trained model](https://drive.google.com/drive/folders/1URpt1Ro1MZhUh-ECdEQFLx0iunlA7K7B?usp=sharing).<br>
3. `python test_pretrained_prediction.py`<br>

## Evaluation
After prediction, users can generate the MSE and macro F1 of low resolution and prediction by running the following command:<br>
`python test_evaluation.py` 

## Heatmap and loss visualization   
If you wish to check the heatmap of low resolution, prediction, and true scHi-C, please run the following command:<br>
`tensorboard --logdir=runs/heatmap` <br>

# Demo
Here we used processed Lee et al. (download from [Download processed data](https://drive.google.com/drive/folders/1EgkzPoNG-s_pi3SKOFG_YFslpIar_Bht?usp=sharing)) to demo the training, prediction and evaluation process:<br>
`>> conda activate SHICEDO` <br>
`> python test_train.py`<br>
`> python test_prediction.py`<br>
`> python test_evaluation.py`<br>
For heatmap and loss visitation: <br>
`tensorboard --logdir=runs/heatmap` 
