### Building the docker container

First, install Docker from the [Docker website](https://docs.docker.com/engine/install/)

When you're ready, download the git registry
```
git clone https://github.com/NeuroSainteAnne/FlairVisibilityVolume.git
cd FlairVisibilityVolume/DockerContainer
```

Place the 10 models (model1 to model10.onnx) into the ONNX_models directory.
Then, you can build a container named `FVV` by typing:
```
docker build -t FVV .
```

### Running the Docker container

You can run the docker container using the following command, and replacing `/my/local/folder` by the folder in which you want to perform inference:

```
docker run --rm --gpus all -v /my/local/folder:/app/data FVV
```

The folder must have at least b0 and b1000 nifti images

```
/my/local/folder
    ├── b0.nii.gz
    └── b1000.nii.gz
```

The inference process will create 4 new files in this folder:
* `predictions.nii.gz`: a NIFTI file with 3 volumes indexed in the 4th dimension: brain mask prediction, stroke area prediction and flair visibility prediction
* `heatmap.nii.gz`: a RGB NIFTI file with a synthesis of these 3 volumes
* `synthesis.png`: a synthesis image displaying slices where a stroke was detected, as well as pertinent scalar values among which the FVV
* `synthesis.txt`: a synthesis text fils with various scalars

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)