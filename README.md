# SHICEDO: Single-cell Hi-C Resolution Enhancement with Reduced Over-smoothing
In this work, we introduce SCHICEDO, a novel deep-learning model specially designed for scHi-C resolu-
tion enhancement while effectively addressing the over-smoothing issue. Based on a Generative Adversarial
Network (GAN) framework, the generator of the SCHICEDO model is designed to handle multi-scale or
varying-size low-resolution scHi-C inputs, ultimately producing an enhanced scHi-C matrix as the output.
Leveraging our group’s previous work. we incorporate and revamped rank-one feature extraction and
reconstruction techniques as well as residual convolution modules, into the SCHICEDO framework.
![Model_Overview](figure/Model_figure.png)
> *This is a citation from a [source](https://www.example.com).*
- [SHICEDO](#SHICEDO:-Single-cell-Hi-C-Resolution-Enhancement-with-Reduced-Over-smoothing)
  - [Installation](#Installation)
  - [Download processed data](#Download-processed-data)
  - [Training](#Training)
  - [Prediction](#Prediction)
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
Please download the processed data to the data folder and use the correct path in the script for data loading.<br>
If you wish to preprocess other datasets. Please check the data preprocessing section

## Data preprocessing
If you wish to process raw data, please run the following command:<br> 
In this example, we show how to process the Nagano et al raw data is available at [Download raw data](https://drive.google.com/drive/folders/1EgkzPoNG-s_pi3SKOFG_YFslpIar_Bht?usp=sharing).<br>
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
We provide two groups of hyper-parameters (in test_train.py) for two processed datasets [Download processed data](https://drive.google.com/drive/folders/1EgkzPoNG-s_pi3SKOFG_YFslpIar_Bht?usp=sharing).<br>
The model and date setting were the same as described in the paper.  <br>
After choosing suitable hyper-parameters, the model can be trained with the following command: <br>
`python test_train.py` <br>

## Prediction
After training, Enhanced scHi-C can predict with the following command:<br>
`python test_prediction.py` <br>
Users can also use the provided pre-trained model to make predictions. <br>
Please change the corresponding model loading path in the test_prediction.py file.<br>

## Evaluation
After prediction, users can generate the MSE and macro F1 of low resolution and prediction by running the following command:<br>
`python test_evaluation.py` 

## Heatmap and loss visualization   
If you wish to check the heatmap of low resolution, prediction, and true scHi-C, please run the following command:<br>
`tensorboard --logdir=runs/train_vali_loss` <br>
`tensorboard --logdir=runs/heatmap` <br>

# Demo
Here we used processed Nagano et al. (download from [Download processed data](https://drive.google.com/drive/folders/1EgkzPoNG-s_pi3SKOFG_YFslpIar_Bht?usp=sharing)) to demo the training, prediction and evaluation process:<br>
`>> conda activate SHICEDO` <br>
`> python test_train.py`<br>
`> python test_prediction.py`<br>
`> python test_evaluation.py`<br>
For heatmap and loss visitation: <br>
`tensorboard --logdir=runs/train_vali_loss` <br>
`tensorboard --logdir=runs/heatmap` 
