# Multiple image joint learning for image aesthetic assessment（MILNet）

This repository contains a pytorch implementation of the paper "Multiple Images Joint Learning for Image Aesthetic Assessment"(Subject to TMM)

We proposed MILNet employ multiple images for aesthetic assessment. First, we adopt semantic information retrieval reference; Then, we use Graph Convolution Network(GCN) to reason the relation of different nodes; Finally, we utilize the improved loss function AdaEMD to stabilize training process.

## Pipeline
![在这里插入图片描述](https://img-blog.csdnimg.cn/0005050d73b4459284644d4d7c232379.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaGVsbG93b3JsZF9GbHk=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## Requirements
- System: **Linux**(e.g. Ubuntu/CentOS/Arch), macOS, or **Windows** Subsystem of Linux (WSL)
- Python version >=3.6
- Pytorch == 1.10.2 Cuda == 10.2 
- TensorboardX
- Opencv == 4.5.5

## Install
- Clone repo
```python
git clone https://github.com/dafeigediaozhatian/MILNet
cd MILNet
```

- Install dependencies(pytorch, scikit-learn, opencv-python, pandas. Recommend to use Anaconda.)
```python
# Create a new conda environment
conda create -n MILNet python==3.8
conda activate MILNet

# Install other packages
pip install -r requirements.txt
```


## Dataset
- AVA dataset(19928)
  - Download the original AVA dataset and dataset split into path_to_AVAdataset_19928/. The directory structure should be like:
	```
	path_to_AVAdataset_19928
		19928_aesthetic_image_list
		AVA_19928.txt
		train_19928.txt
		test_19998.txt
	```

- AVA dataset(19998)
   - Download the dataset split into path_to_AVAdataset_19998/. The directory structure should be like:
		```
		path_to_AVAdataset_19998
			19998_aesthetic_image_list
			AVA_19998.txt
			train_19998.txt
			test_19998.txt
		```
- Extract feature
We extract the AVA dataset to MLSP, and feature reduce to 6144 for training. all the files can be download on [here](https://pan.baidu.com/s/1j02Of7k5_6rQQMqOaI6I3g),code is #ddkd#.



## Training
Traning scripts for two datasets can be found in main/. The dataroot argument should be modified to path_to_<dataset_name>. Run the follwing command for training:
```python
# Training on AVA_19998
python main/train_AVA_19928.py

# Training on AVA_19928
python main/train_AVA_19998.py
```



## Testing
The [pre-trained model]() that predict distribution directly from images
Testing model by runing the scripts or the follwing command:
```python
python -m test_AVA_19928.py \
    --dataset <dataset_name> \
    --dataroot path_to_<dataset_name> \
    --eval_model path_to_model
    
python -m test_AVA_19998.py \
    --dataset <dataset_name> \
    --dataroot path_to_<dataset_name> \
    --eval_model path_to_model
```


## Citation
