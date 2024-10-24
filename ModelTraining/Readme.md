# Data Analysis Pipeline

This repository contains a series of Python scripts designed for processing, training, model conversion, and testing in the context of medical imaging analysis. Below is a comprehensive description of each script in the pipeline, explaining their specific purpose and the overall workflow.

## Table of Contents
* [Environment](#0-environment)
* [1-DataPreparation.py](#1-data-preparation)
* [2a-Training.py](#2a-training)
* [2b-TrainingKFolds.py](#2b-training-k-folds)
* [3-ConvertModels.py](#3-convert-models)
* [4-InferenceTest.py](#4-inference-test)

## 0. Environment

In order to use these script, you can create a tailored conda environment.

First, clone the whole git repository and go to ModelTraining folder

```
git clone https://github.com/NeuroSainteAnne/FlairVisibilityVolume.git
cd FlairVisibilityVolume/ModelTraining
```

You can now create and activate a conda environment (tested with conda 24.9.2)

```
conda env create -f environment.yml
conda activate FVVenv
```

## 1. Data Preparation

### Script: `1-DataPreparation.py`

The `1-DataPreparation.py` script is the starting point of the entire pipeline. It focuses on preparing the input data, which includes loading, preprocessing, and organizing the dataset for model training and testing. Key tasks include:

- **Loading Input Images**: This script reads input medical imaging data such as b0 and b1000 images.
- **Normalization and Rescaling**: Each image is normalized and rescaled to ensure consistency in size and pixel intensity across all images.
- **Mask Generation**: If a brain mask is not provided, the script generates a mask using an Otsu threshold or other specified methods.
- **Data Saving**: The processed data is saved in a memory-efficient format to facilitate faster training and testing in subsequent steps.

The input data should be organized in an `input_data` folder with the following structure:
```
input_data/
    ├── Patient_1/
    │     ├── b0.nii.gz
    │     ├── b1000.nii.gz
    │     ├── stroke_roi.nii.gz : large binary ROI over the stroke area
    │     ├── flairviz_roi.nii.gz : regional delineation of FLAIR visible areas
    │     └── (optional) mask.nii.gz : brain mask (will be computed if necessary
    ├── Patient_2/
    │     ├── b0.nii.gz
    │     ├── b1000.nii.gz
    │     ├── stroke_roi.nii.gz 
    │     ├── flairviz_roi.nii.gz 
    │     └── (optional) mask.nii.gz
    └── ...
```

Usage:

```
python 1-DataPreparation.py  [-k KFOLD] [-v VALID_PROP] 

Arguments:
   -k, --kfolds           : Number of kfolds (default: 10)
   -v, --valid_proportion : Proportion of valid/train in each fold (default: 0.2)
```

## 2a. Training

### Script: `2a-Training.py`

The `2a-Training.py` script is responsible for training a neural network using the processed data from the data preparation stage. Key features of this script include:

- **Model Initialization**: The script initializes the model architecture and sets up the optimizer, loss functions, and metrics.
- **Training Loop**: It includes a training loop where batches of data are fed into the model, and the weights are updated based on the computed loss.
- **Logging**: During training, metrics such as training loss and accuracy are logged for monitoring purposes on Weights and Biases, and checkpoints of the model are saved for evaluation.
  
Usage:
```
python 2a-Training.py [-k KFOLD] [-n NAME] [-d DEBUG] [-ram PRELOAD_RAM] [-w WANDB_PROJECT]

Arguments:
   -k, --kfold           : Fold index for k-fold cross-validation (default: 0)
   -n, --name            : Name for the experiment/run (default: "FVV")
   -d, --debug           : Debug mode, set to enable debugging (default: False)
   -ram, --preload_RAM   : Preload data into RAM, set to enable (default: False)
   -w, --wandb_project   : Weights & Biases project name for logging (default: "FVV")
```

## 2b. Training K-Folds

### Script: `2b-TrainingKFolds.py`

The `2b-TrainingKFolds.py` script extends the standard training process by introducing **K-fold Cross-Validation**, which is a robust method to validate model performance:

- **K-Fold Splitting**: The dataset is divided into multiple folds (K). In each iteration, one fold is used for validation while the others are used for training.
- **Model Training and Validation**: The model is trained K times, each time with a different validation set. This helps in assessing the generalization capability of the model more accurately.
- **Model Checkpointing**: The script saves the model from each fold, allowing the user to compare model performances across different splits.

Usage:

```
python 2b-TrainingKFolds.py [-n NAME] [-ram PRELOAD_RAM] [-w WANDB_PROJECT]

Arguments:
   -n, --name            : Name for the experiment/run (default: "FVV")
   -ram, --preload_RAM    : Preload data into RAM, set to enable (default: False)
   -w, --wandb_project    : Weights & Biases project name for logging (default: "FVV")
```

## 3. Convert Models

### Script: `3-ConvertModels.py`

The `3-ConvertModels.py` script takes the trained models and converts them into a format suitable for inference, such as ONNX:

- **Model Loading**: It loads the model checkpoints from the training stage.
- **ONNX Conversion**: Converts the PyTorch models into ONNX format, which is widely used for deployment and inference.
- **Exporting**: The converted models are saved to a specified directory, ready for deployment in the inference script.

The ONNX models produced by this script are more portable and can be used for accelerated inference on various platforms.

Usage :

```
First, modify the script to set 1 desired model per each cross-validation fold.

Then, launch the script:
python 3-ConvertModels.py
```

## 4. Inference Test

### Script: `4-InferenceTest.py`

The `4-InferenceTest.py` script runs the inference process using the converted ONNX models to evaluate their performance:

- **Model Inference**: Loads the ONNX models and performs inference on the test data to generate predictions.
- **Post-Processing**: Processes the raw outputs from the model, such as applying thresholding to create binary masks.
- **Metrics Calculation**: Calculates performance metrics, such as accuracy, Dice coefficient, and others, to assess the quality of predictions.
- **Heatmap Generation**: Optionally, generates visual heatmaps of the model predictions for evaluation purposes.

Usage:

```
python 4-InferenceTest.py
```

