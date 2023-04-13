conda create -n py36 python=3.6
conda activate py36

conda install -y pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
conda install -y -c menpo opencv
conda install -y -c anaconda scipy

CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./logs/

CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
 --dataset_file SHHA \
 --epochs 100 \
 --lr_drop 3500 \
 --output_dir ./logs \
 --checkpoints_dir ./weights \
 --tensorboard_dir ./logs \
 --lr 0.0001 \
 --lr_backbone 0.00001 \
 --batch_size 8 \
 --eval_freq 1 \
 --gpu_id 0

testing result:

```
vgg11_bn:
vgg13_bn:
vgg16_bn: 2.13022 sec

```

video
CUDA_VISIBLE_DEVICES=0 python run_video.py --weight_path ./weights/SHTechA.pth
