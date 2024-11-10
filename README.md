# Few-shot Stereo Matching with High Domain Adaptability Based on Adaptive Recursive Network

New: Our testing code is currently under development and will be released soon.

This repository contains the code for the paper *"Few-shot Stereo Matching with High Domain Adaptability Based on Adaptive Recursive Network"*. The training code is now released. Our testing code will be publicly available soon.

## Getting Started

### Step 1: Prepare the Training Data

To begin, run the `get_training_data.py` script. This script will randomly sample image pairs from the `SceneFlow` directory and create the training dataset. The images will be saved in the `training_patchs` folder.

> **Note**: Use `--root_dir /path/to/your/SceneFlow/` to specify your `SceneFlow` directory.

### Step 2: Prepare the Testing Data

Similarly, run the `get_testing_data.py` script to sample image pairs for the testing dataset. The images will be saved in the `testing_patch` folder.

> **Note**: Use `--root_dir /path/to/your/SceneFlow/` to specify your `SceneFlow` directory.

### Step 3: Train the Model

After generating the training and testing datasets, you can start training the model by running `train.py`. This script will read the training and testing data from the `training_patchs` and `testing_patch` folders, respectively. The trained model will be saved in the `good_model` directory.

## Directory Structure

- `SceneFlow/`: Root directory containing the input image files.
- `training_patchs/`: Directory where sampled training image pairs are stored.
- `testing_patch/`: Directory where sampled testing image pairs are stored.
- `good_model/`: Directory where the trained model is saved.

## Example Usage

1. Run `get_training_data.py`:

   ```bash
   python get_training_data.py --root_dir /path/to/your/SceneFlow/ --num_processes 4 --images_num 500
   ```

2. Run `get_testing_data.py`:

   ```bash
   python get_testing_data.py --root_dir /path/to/your/SceneFlow/ --num_processes 4 --images_num 500
   ```

3. Run `train.py` to train the model:

   ```bash
   python train.py
   ```

Now, you are ready to train the model and achieve high domain adaptability with few-shot learning!

