# Few-shot Stereo Matching with High Domain Adaptability Based on Adaptive Recursive Network

New: Our testing code is currently under development and will be released soon.

This repository contains the code for the paper *"Few-shot Stereo Matching with High Domain Adaptability Based on Adaptive Recursive Network"*. The training code is now released. Our testing code will be publicly available soon.

## Getting Started and Example Usage

### Step 1: Prepare the Training Data

To begin, run the `get_training_data.py` script. This script will randomly sample image pairs from the `SceneFlow` directory and create the training dataset. The images will be saved in the `training_patchs` folder.

> **Note**: Use `--root_dir /path/to/your/SceneFlow/` to specify your `SceneFlow` directory.

Run the script:

```bash
python get_training_data.py --root_dir /path/to/your/SceneFlow/ --num_processes 4 --images_num 500
```

### Step 2: Prepare the Testing Data

Similarly, run the `get_testing_data.py` script to sample image pairs for the testing dataset. The images will be saved in the `testing_patch` folder.

> **Note**: Use `--root_dir /path/to/your/SceneFlow/` to specify your `SceneFlow` directory.

Run the script:

```bash
python get_testing_data.py --root_dir /path/to/your/SceneFlow/ --num_processes 4 --images_num 500
```

### Step 3: Train the Model

After generating the training and testing datasets, you can start training the model by running `train.py`. This script will read the training and testing data from the `training_patchs` and `testing_patch` folders, respectively. The trained model will be saved in the `good_model` directory.

Run the training script:

```bash
python train.py --epoch 500
```

### Training Script Command-line Arguments

The `train.py` script supports the following command-line argument:

- `--epoch`: Specifies the number of epochs to train the model (default is 500).

## Directory Structure

- `SceneFlow/`: Root directory containing the input image files.
- `training_patchs/`: Directory where sampled training image pairs are stored.
- `testing_patch/`: Directory where sampled testing image pairs are stored.
- `good_model/`: Directory where the trained model is saved.


