# FlairVisibilityVolume

## Scientific Background: FVV

**FLAIR Visibility Volume (FVV)** is a quantitative biomarker that helps in assessing DWI-FLAIR mismatch, a key concept in acute ischemic stroke (AIS) management. DWI-FLAIR mismatch is used to estimate whether a stroke patient may be within the therapeutic window for treatment, especially when the exact time of stroke onset is unknown. FVV is derived *solely from Diffusion-Weighted Imaging (DWI)* data and serves as a surrogate for the DWI-FLAIR mismatch, offering a more objective and automated assessment.

FVV has been trained on a large, multicenter dataset (ETIS) and validated using an independent dataset (WAKE-UP trial), demonstrating high diagnostic accuracy. By using only DWI data, FVV aims to reduce dependency on FLAIR sequences, which are often affected by motion artifacts and may not be available in some emergency situations. The models in this repository are designed to predict FVV, providing an efficient and reliable tool for guiding treatment decisions in AIS patients.

## Repository Structure

The repository contains two main folders:

### 1. Model Training

The [Model Training](ModelTraining) folder includes Python scripts for training deep learning models and performing inference on medical imaging data. The training scripts allow for the preparation, training, and evaluation of neural networks designed for medical image segmentation and analysis. The main functionalities provided include:

- **Data Preparation**: Preprocessing of medical images, including normalization and rescaling.
- **Model Training**: Training scripts that utilize advanced machine learning techniques, including K-fold cross-validation, for robust model evaluation.
- **Model Inference**: Scripts for applying the trained models to test data to generate predictions and assess performance.
- **Direct Inference**: For direct inference on NIFTI files.

### 2. Docker Container

The [Docker Container](DockerContainer) folder provides a Docker setup for fast inference using pre-trained models. The Docker container allows users to efficiently run the model inference process without needing to install complex dependencies manually. It is ideal for deployment in production environments or for sharing the inference capabilities with collaborators.

#### Example of synthesis map created with FVV Docker Container
<img src="images/synthesis.png?raw=true" width="500" alt="Example of synthesis map">
The DWI lesion was also visible on FLAIR sequence (no DWI-FLAIR mismatch)

### 3. Model weights

The complete code with model weights, whose training is described in the original paper, can be downloaded in the Releases section.

## Getting Started

To get started with this repository, clone the repository and follow the setup instructions provided in the README files of the respective folders (`ModelTraining` and `DockerContainer`). Detailed setup instructions for creating the appropriate environment are provided.

## License
This repository is licensed under the ??? License. Please see the `LICENSE` file for more details.

