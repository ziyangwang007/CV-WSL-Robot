## CV-WSL-Robot
Exploring CNN and ViT for Weakly-Supervised Surgical Tool Segmentation

## Demo Picture

<img src="demodata.png">


## Requirements
* [Pytorch]
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy, Medpy ......

## Datasets
Release soon after acceptance

## Baseline Methods
For other Weakly-Supervised segmentation methods -> [CV-WSL-MIS](https://github.com/ziyangwang007/CV-WSL-MIS)


## Usage

1. Clone the repo:
```
git clone https://github.com/ziyangwang007/CV-WSL-Robot.git
cd CV-WSL-Robot
```
2. Download the pre-processed data 

The preprocessed data can be downloaded from [Baidu Netdisk Link](https://pan.baidu.com/s/13RJouF0cmTXWxRosk__lUA) with passcode: 'bmrr', or [Google Drive Link]- TBC.



3. Train the model

```
cd code

python train_Ours_Weakly_Consistency_Robot_2D.py 
```

4. Test the model

```
python test_2D_fully_ViT.py

or 

python test_2D_fully.py

```

## Reference
