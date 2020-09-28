You can download pre-trained models from <https://drive.google.com/drive/folders/1LQnVHb5Xo6fKknUiOa1fXmGI_MCucGTC?usp=sharing>
we have tested two backbones, i.e., resnetv1 and resnetv1b, results show that there is no significant difference.

Before beginning , you need to follow <https://github.com/wkcn/MobulaOP.git> to install MobulaOP and setup the COCO2017 dataset.

For training, please use the following command:
```bash
OMP_NUM_THREADS=4 PYTHONPATH=. /home/ubuntu/anaconda3/envs/mxnet_latest_p37/bin/python scripts/train_fcos.py \
--dataset-type=coco --dataset-root=/dev/shm/coco/ --num-classes=81 --config=configs/fcos/fcos_rensetv1b.yaml \
--extra-flag=resnetv1 --nvcc=/usr/local/cuda-10.1/bin/nvcc --im-per-gpu=4
```

For testing, please use the following command:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=. /home/ubuntu/anaconda3/envs/mxnet_latest_p37/bin/python scripts/train_fcos.py \
--dataset-type=coco --dataset-root=/dev/shm/coco/ --num-classes=81 --config=configs/fcos/fcos_rensetv1b.yaml \
--extra-flag=resnetv1 --nvcc=/usr/local/cuda-10.1/bin/nvcc --im-per-gpu=4 --demo --demo-params="6.params"
```

For visualize the testing results on COCO2017, please use the following command:
```bash
OMP_NUM_THREADS=4 PYTHONPATH=. /home/ubuntu/anaconda3/envs/mxnet_latest_p37/bin/python scripts/train_fcos.py \
--dataset-type=coco --dataset-root=/dev/shm/coco/ --num-classes=81 --config=configs/fcos/fcos_rensetv1b.yaml \
--extra-flag=resnetv1 --nvcc=/usr/local/cuda-10.1/bin/nvcc --im-per-gpu=4 --demo --demo-params="6.params" --viz
```

