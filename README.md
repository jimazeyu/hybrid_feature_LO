# hybrid_feature_LO
hybrid_feature_LO

## unit_operations

- [kitti_datasets](https://yunpan.tongji.edu.cn/link/AAABC09F5A03104CF6A3C360DE77FD9B7A)

download and put in ./unit_operations/datasets

## Rangenet++
### dataset
- [SemanticKITTI](http://semantic-kitti.org)

download KITTI Odemetry Benchmark Velodyne point clouds (80 GB) & SemanticKITTI label data (179MB)

put in ./Rangenet++/train/tasks/semantic/dataset

### Pre-trained Models
more details in Rangenet++/README.md

download and put in ./Rangenet++/train/tasks/semantic/pre_trained_model

### Order
```
cd Rangenet++/train/tasks/semantic
./train.py -d ./dataset -ac ./config/arch/darknet21.yaml -l ./log 
```

if use pre-trained model

e.x.
```
./train.py -d ./dataset -ac ./config/arch/darknet21.yaml -l ./log -m ./pre_trained_model/darknet21
```