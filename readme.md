# AI for medical imaging — Fall 2024 course project

## Codebase use

### Setting up the environment
```
git clone https://github.com/raoulritter/ai4mi_project.git
cd ai4mi_project
git submodule init
git submodule update
```

This codebase was written for a somewhat recent python (3.10 or more recent). (**Note: Ubuntu and some other Linux distributions might make the distasteful choice to have `python` pointing to 2.+ version, and require to type `python3` explicitly.**) The required packages are listed in [`requirements.txt`](requirements.txt) and a [virtual environment](https://docs.python.org/3/library/venv.html) can easily be created from it through [pip](https://pypi.org/):
```
python -m venv ai4mi
source ai4mi/bin/activate
which python  # ensure this is not your system's python anymore
python -m pip install -r requirements.txt
```
Conda is an alternative to pip, but is recommended not to mix `conda install` and `pip install`.

### Getting the data
The first thing that is important is to download the [train and validation data](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EfMdFte7pExAnPwt4tYUcxcBbJJO8dqxJP9r-5pm9M_ARw?e=ZNdjee) and the [test data](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EWZH7ylUUFFCg3lEzzLzJqMBG7OrPw1K4M78wq9t5iBj_w?e=Yejv5d), both of which require a UvA account. Rename both files to `segthor_train.zip` and `test.zip` and place them inside the `data` folder. The following two commands prepare the data in the correct way.
```
make data/SEGTHOR
make data/SEGTHOR_TEST
```

### Preprocessing
Whether you want to use preprocessing or not, this is a step that is always useful to execute, just to have the data available. This allows you to later toggle preprocessing on or off very easily. In our paper the full description of preprocessing can be found, along with an explanation of why it is highly recommended. Run the following commands to perform preprocessing on the input data. We have an additional preprocess file for the test set. 
```
python run_preprocess.py
python run_test_preprocess.py
```
The preprocessed data is outputted in the `data/SEGTHOR_preprocessed` folder. If you are interested in the reasons why we perform these preprocessing you can run the following commands:
```
cd preprocessing
python voxel_intensity.py
python voxel_size.py
cd ..
```
In the `preprocessing/insights` folder you will now find two logs, showing you both the pixel intensities and voxel sizes of the original scans. More details about interpretation of these can be found in our paper.

### Data Augmentation
In our researched we used offline data augmentation in order to expand our dataset and attempt to improve the quality of our predictions. Similar to preprocessing this step is useful to do in advance, to enable the simple on or off 'toggling' of data augmentation later. The following command performs random augmentations on the data and in doing so double the dataset in size:
```
python run_augmentation.py
```
**Note (not recommended):** This data augmentation runs the augmentation on the already preprocessed data. It is highly recommended to do this, because the original data (due to a transformation in the ground truth heart segmentation) does not make too much sense. If you still want to perform the data_augmentation on the original data you can run:
```
python run_augmentation.py --slice_dir data/SEGTHOR --dest_dir data/SEGTHOR
```

### Model Training
Now we get to the most important part; model training. How do we train our model? The core file to do this is `main.py`. However, there are multiple arguments that can be passed here which cause the model to be trained with different settings. Here is a brief overview of these arguments:
* `--model_name`: Perhaps the most important argument. Which model architecture do you want to train here? There are three options: `'ENet'` (the baseline model), `'SAM2'` or `'VMUNet'`.
* `--epochs`: Allows you to enter the amount of epochs you want to train the model for. All our experiments were run using 50 epochs.
* `-O`: The codebase uses a lot of assertions for control and self-documentation, this be disabled with this `-O` option (for faster training).
* `--preprocess`: Setting this flag means you are using the preprocessed data instead of the raw input data.
* `--augmentation`: Setting this flag means you are expanding the input dataset in size by using additional augmented data. This argument can only be set when the `--preprocess` argument is also set.
* `--tuning`: When you set this flag, we are using different options for the baseline architecture, such as a learning rate schedule, weight decay and a different optimizer. Tuning can only be set when the `--model_name` is set to `'ENet'`.
* `--mode`: This can be either `'partial'` or `'full'` (train on part of the classes or all the classes, the specific classes can be set in `main.py`.
* `--gpu`: Use a GPU if available.
* `--dest`: Sets a destination folder for the results

To start our experiment and set a **baseline** we run the full network using the raw input data and baseline model as we received it from the course coordinators. This can be done using the following command:
```
python -O main.py --model_name ENet --dataset SEGTHOR --mode full --epochs 50 --dest results/SEGTHOR/ce --gpu
```
In the table below you can see a brief overview of all the different model settings that have been tested for this research. More details can be found in our paper.

| Settings | ENet     | VM-Unet  | SAM2     |
|----------|----------|----------|----------|
| Baseline | ✔| ✔     | ✔ | 
| Preprocessing |✔|✔|✔|
| Preprocessing + Augmentation    | ✔ | ✔    | ✔  |
| Preprocessing + Tuning    | ✔    |   ✖   |    ✖  |
| Preprocessing + Post-processing    | ✔     | ✔     | ✔     |
| Preprocessing + Augmentation + Post-Processing    | ✔     | ✔     | ✔     |
| Preprocessing + Augmentation + Tuning   | ✔     | ✖     | ✖     |
| Preprocessing + Tuning + Post-Processing   | ✔     | ✖     | ✖     |
| Preprocessing + Augmentation + Tuning + Post-Processing   | ✔     | ✖     | ✖     |

### Post-processing
Since post-processing is a relatively cheap operation, it is included in the standard model training, however both the raw network output and the post-processed output are saved. Metrics calculation is also performed on both outputs in order to be able to compare the two results. In later versions it will be possible to disable post-processing using a post-processing flag.

### Results
The results can be viewed in multiple ways. Most conveniently a zip file with all necessary results and log is also create for you. 

### SAM2 Experiments
Note: The SAM2 experiments and configurations are maintained in the 'raoul' branch of this repository. These have been kept separate from the main branch due to:
- Additional configuration requirements
- Extra hyperparameter settings
- Specialized environment dependencies
- Memory-intensive operations

To access and run the SAM2 experiments:
1. Switch to the raoul branch: `git checkout raoul`
2. Follow the extended setup instructions in the branch's README
3. Ensure you have sufficient computational resources available
