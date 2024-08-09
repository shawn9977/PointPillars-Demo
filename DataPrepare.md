##### Test Env

- ubuntu 20.04
- python 3.8
- cuda version 12.2

##### 1. Create and enter python virtual environment

```shell
python -m venv venv
source ./testenv/bin/activate
```

##### 2. Pip install package

```shell
pip install --upgrade pip
pip install shapely pybind11 protobuf scikit-image numba pillow
pip install torch torchvision torchaudio
pip install fire tensorboardX
```

3. ##### Apt install package

```shell
apt install libboost-all-dev
```

4. ##### Export

```shell
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

5. ##### Prepare original dataset and directory structure

Prepare the data directory with the following structure

Download the original dataset and put it in the corresponding directory. Download address:

- website： https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
- other
    - image_2 https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
    - calib https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
    - label_2 https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
    - velodyne https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip

```
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

6. ##### Prepare project

```shell
git clone https://github.com/SmallMunich/nutonomy_pointpillars.git
export PYTHONPATH=$PYTHONPATH:/your_root_path/nutonomy_pointpillars/
vim ./nutonomy_pointpillars/second/core/cc/nms/nms_cpu.cpp
#add in nms_cpu.cpp: #include <iostream>
```

7. ##### Process the original dataset

```shell
cd ./nutonomy_pointpillars/second
```

- Create kitti infos:

  ```shell
  python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
  ```

- Create reduced point cloud:

  ```shell
  python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
  ```

- Create groundtruth-database infos:

  ```shell
  python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
  ```