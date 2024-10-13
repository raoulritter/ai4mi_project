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
Whether you want to use preprocessing or not, this is a step that is always useful to execute, just to have the data available. This allows you to later toggle preprocessing on or off very easily. In our paper the full description of preprocessing can be found, along with an explanation of why it is highly recommended. Run the following commands to perform preprocessing on the input data.
```
python run_preprocess.py
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
| Baseline | ✔|  | ✔     | ✔ | 
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




[open text]




## Project overview
The project is based around the SegTHOR challenge data, which was kindly allowed by Caroline Petitjean (challenge organizer) to use for the course. The challenge was originally on the segmentation of different organs: heart, aorta, esophagus and trachea.
![Segthor Overview](segthor_overview.png)

## Codebase features
This codebase is given as a starting point, to provide an initial neural network that converges during training. (For broader context, this is itself a fork of an [older conference tutorial](https://github.com/LIVIAETS/miccai_weakly_supervised_tutorial) we gave few years ago.) It also provides facilities to locally run some test on a laptop, with a toy dataset and dummy network.

Summary of codebase (in PyTorch)
* slicing the 3D Nifti files to 2D `.png`;
* stitching 2D `.png` slices to 3D volume compatible with initial nifti files;
* basic 2D segmentation network;
* basic training and printing with cross-entroly loss and Adam;
* partial cross-entropy alternative as a loss (to disable one class during training);
* debug options and facilities (cpu version, "dummy" network, smaller datasets);
* saving of predictions as `.png`;
* log the 2D DSC and cross-entropy over time, with basic plotting;
* tool to compare different segmentations (`viewer/viewer.py`).

**Some recurrent questions might be addressed here directly.** As such, it is expected that small change or additions to this readme to be made.

## Codebase use
In the following, a line starting by `$` usually means it is meant to be typed in the terminal (bash, zsh, fish, ...), whereas no symbol might indicate some python code.

### Setting up the environment
```
$ git clone https://github.com/raoulritter/ai4mi_project.git
$ cd ai4mi_project
$ git submodule init
$ git submodule update
```

This codebase was written for a somewhat recent python (3.10 or more recent). (**Note: Ubuntu and some other Linux distributions might make the distasteful choice to have `python` pointing to 2.+ version, and require to type `python3` explicitly.**) The required packages are listed in [`requirements.txt`](requirements.txt) and a [virtual environment](https://docs.python.org/3/library/venv.html) can easily be created from it through [pip](https://pypi.org/):
```
$ python -m venv ai4mi
$ source ai4mi/bin/activate
$ which python  # ensure this is not your system's python anymore
$ python -m pip install -r requirements.txt
$ pre-commit install
```
Conda is an alternative to pip, but is recommended not to mix `conda install` and `pip install`.

### Getting the data
The synthetic dataset is generated randomly, whereas for Segthor it is required to put the file [`segthor_train.zip`](https://amsuni-my.sharepoint.com/:u:/g/personal/h_t_g_kervadec_uva_nl/EfMdFte7pExAnPwt4tYUcxcBbJJO8dqxJP9r-5pm9M_ARw?e=ZNdjee) (required a UvA account) in the `data/` folder. If the computer running it is powerful enough, the recipe for `data/SEGTHOR` can be modified in the [Makefile](Makefile) to enable multi-processing (`-p -1` option, see `python slice_segthor.py --help` or its code directly).
```
$ make data/TOY2
$ make data/SEGTHOR
```

For windows users, you can use the following instead
```
$ rm -rf data/TOY2_tmp data/TOY2
$ python gen_two_circles.py --dest data/TOY2_tmp -n 1000 100 -r 25 -wh 256 256
$ mv data/TOY2_tmp data/TOY2

$ sha256sum -c data/segthor_train.sha256
$ unzip -q data/segthor_train.zip

$ rm -rf data/SEGTHOR_tmp data/SEGTHOR
$ python  slice_segthor.py --source_dir data/segthor_train --dest_dir data/SEGTHOR_tmp \
         --shape 256 256 --retain 10
$ mv data/SEGTHOR_tmp data/SEGTHOR
````

### Viewing the data
The data can be viewed in different ways:
- looking directly at the `.png` in the sliced folder (`data/SEGTHOR`);
- using the provided "viewer" to compare segmentations ([see below](#viewing-the-results));
- opening the Nifti files from `data/segthor_train` with [3D Slicer](https://www.slicer.org/) or [ITK Snap](http://www.itksnap.org).

### Training a base network
Running a training
```
$ python main.py --help
usage: main.py [-h] [--epochs EPOCHS] [--dataset {TOY2,SEGTHOR}] [--mode {partial,full}] --dest DEST [--gpu] [--debug]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS
  --dataset {TOY2,SEGTHOR}
  --mode {partial,full}
  --dest DEST           Destination directory to save the results (predictions and weights).
  --gpu
  --debug               Keep only a fraction (10 samples) of the datasets, to test the logic around epochs and logging easily.
$ python main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

The codebase uses a lot of assertions for control and self-documentation, they can easily be disabled with the `-O` option (for faster training) once everything is known to be correct (for instance run the previous command for 1/2 epochs, then kill it and relaunch it):
```
$ python -O main.py --dataset TOY2 --mode full --epoch 25 --dest results/toy2/ce --gpu
```

### Viewing the results
#### 2D viewer
Comparing some predictions with the provided [viewer](viewer/viewer.py) (right-click to go to the next set of images, left-click to go back):
```
$ python viewer/viewer.py --img_source data/TOY2/val/img \
    data/TOY2/val/gt results/toy2/ce/iter000/val results/toy2/ce/iter005/val results/toy2/ce/best_epoch/val \
    --show_img -C 256 --no_contour
```
![Example of the viewer on the TOY example](viewer_toy.png)
**Note:** if using it from a SSH session, it requires X to be forwarded ([Unix/BSD](https://man.archlinux.org/man/ssh.1#X), [Windows](https://mobaxterm.mobatek.net/documentation.html#1_4)) for it to work. Note that X forwarding also needs to be enabled on the server side.


```
$ python viewer/viewer.py --img_source data/SEGTHOR/val/img \
    data/SEGTHOR/val/gt results/segthor/ce/iter000/val results/segthor/ce/best_epoch/val \
    -n 2 -C 5 --remap "{63: 1, 126: 2, 189: 3, 252: 4}" \
    --legend --class_names background esophagus heart trachea aorta
```
![Example of the viewer on SegTHOR](viewer_segthor.png)

#### 3D viewers
To look at the results in 3D, it is necessary to reconstruct the 3D volume from the individual 2D predictions saved as images.
To stitch the `.png` back to a nifti file:
```
$ python stitch.py --data_folder results/segthor/ce/best_epoch/val \
    --dest_folder volumes/segthor/ce \
    --num_classes 255 --grp_regex "(Patient_\d\d)_\d\d\d\d" \
    --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"
```

[3D Slicer](https://www.slicer.org/) and [ITK Snap](http://www.itksnap.org) are two popular viewers for medical data, here comparing `GT.nii.gz` and the corresponding stitched prediction `Patient_01.nii.gz`:
![Viewing label and prediction](3dslicer.png)

Zooming on the prediction with smoothing disabled:
![Viewing the prediction without smoothing](3dslicer_zoom.png)


### Plotting the metrics
There are some facilities to plot the metrics saved by [`main.py`](main.py):
```
$ python plot.py --help
usage: plot.py [-h] --metric_file METRIC_MODE.npy [--dest METRIC_MODE.png] [--headless]

Plot data over time

options:
  -h, --help            show this help message and exit
  --metric_file METRIC_MODE.npy
                        The metric file to plot.
  --dest METRIC_MODE.png
                        Optional: save the plot to a .png file
  --headless            Does not display the plot and save it directly (implies --dest to be provided.
$ python plot.py --metric_file results/segthor/ce/dice_val.npy --dest results/segthor/ce/dice_val.png
```
![Validation DSC](dice_val.png)

## Submission and scoring
Groups will have to submit:
* archive of the git repo with the whole project (pre-processing/training/post-processing where applicable, inference and metrics);
* the best trained model;
* predictions on the test set (will be shared later on);
* predictions on the group's internal validation set, validation set, and the metrics they computed.

The main criteria for scoring will include:
* improvement of performances over baseline;
* code quality/clear [git use](git.md);
* the [choice of metrics](https://metrics-reloaded.dkfz.de/);
* correctness of the computed metrics (on the validation set);
* (part of the report) clear description of the method;
* report.


### Packing the code
`$ git bundle group-XX.bundle master`

### Saving the best model
`torch.save(net, args.dest / "bestmodel-group-XX.pkl")`

### Archiving everything for submission
All files should be grouped in single folder with the following structure
```
group-XX/
    test/
        pred/
            Patient_41.nii.gz
            Patient_42.nii.gz
            ...
    val/
        pred/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        gt/
            Patient_21.nii.gz
            Patient_32.nii.gz
            ...
        metric01.npy
        metric02.npy
        ...
    group-XX.bundle
    bestmodel-group-XX.pkl
```
The metrics should be numpy `ndarray` with the shape `NxKxD`, with `N` the number of scan in the subset, `K` the number of classes (5, including background), and `D` the eventual dimensionality of the metric (can be simply 1).

The folder should then be [tarred](https://xkcd.com/1168/) and compressed, e.g.:
```
$ tar cf - group-XX/ | zstd -T0 -3 > group-XX.tar.zst
$ tar cf group-XX.tar.gz - group-XX/
```


## Known issues
### Cannot pickle lambda in the dataloader
Some installs (probably due to Python/Pytorch version mismatch) throw an error about an inability to pickle lambda functions (at the dataloader stage). Short of reinstalling everything, setting the number of workers to 0 seems to get around the problem (`--num_workers 0`).

### Pytorch not compiled for Numpy 2.0
It may happen that Pytorch, when installed through pip, was compiled for Numpy 1.x, which creates some inconsistencies. Downgrading Numpy seems to solve it: `pip install --upgrade "numpy<2"`

### Viewer on Windows
Windows has different paths names (`\` in stead of `/`), so the default regex in the viewer needs to be changed to `--id_regex=".*\\\\(.*).png"`.
