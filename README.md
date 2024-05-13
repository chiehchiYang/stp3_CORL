# st-p3_region

## Setup the Environment 
```bash
conda env create -f environment.yml

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

conda install pytorch-lightning -c conda-forge

pip install moviepy
```



bash scripts/train_perceive.sh /home/chiehchiyang/Desktop/ST-P3/stp3/configs/carla/Perception.yml /media/chiehchiyang/WD/dataset

## Train prediction 

``` bash

conda activate carla_tfuse
bash scripts/train_perceive.sh ./stp3/configs/carla/Perception.yml /media/chiehchiyang/WD/dataset_my
########################################################
bash scripts/train_prediction.sh ./stp3/configs/carla/Prediction.yml /media/chiehchiyang/WD/dataset_my ./checkpoint/stage1.ckpt
#######################################################
bash scripts/eval_plan.sh ./epoch\=6.ckpt /media/chiehchiyang/WD/dataset 
tensorboard --logdir=./tensorboard_logs/
screen -S chiehchi
ctrl + A, D # 記法：detach
screen -r <screen_id> 
```


### Experiments 

#### Goal area
- train with semantic segmentation head 
- train with heatmap head 
