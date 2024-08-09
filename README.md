# PointPillars  Demo

Archive: No
Date: July 30, 2024
Status: Not started

## Notes

<aside>
<img src="https://www.notion.so/icons/document_gray.svg" alt="https://www.notion.so/icons/document_gray.svg" width="40px" />

</aside>

<aside>
<img src="https://www.notion.so/icons/document_gray.svg" alt="https://www.notion.so/icons/document_gray.svg" width="40px" />

</aside>

---

## To Do’s

---

- [ ]  
- [ ]  
- [ ]  

---

在 MTL 上  openvino  为   2024.3.0进行PointPillars Demo 展示

# 环境配置说明

## **硬件：**

 **一台集成iGPU的MTL**

## 软件：

 系统信息

- **内核版本**: 6.5.0-45-generic
- **操作系统**: Ubuntu 22.04.4 LTS (Jammy)
- **CPU**:
    - **架构**: x86_64
    - **型号**: Intel(R) Core(TM) Ultra 7 165H
    - **核心数量**: 16
    - **线程数量**: 32
    - **最大频率**: 4900 MHz
    - **最小频率**: 400 MHz

## 一 安装与配置步骤

### 1.安装iGPU的驱动

安装的Computer runtime 版本是**24.26.30049.6**

参考安装方法：
https://github.com/intel/compute-runtime/releases/tag/24.26.30049.6

```python

安装结束后 创建个 python的虚拟环境 ， 安装openvino-dev 
然后查看下 benchmark_app -h 是否有GPU

(test_env) shawn@mtl:~/OpenPCDet/tools$ benchmark_app -h
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
usage: benchmark_app [-h [HELP]] [-i PATHS_TO_INPUT [PATHS_TO_INPUT ...]] -m PATH_TO_MODEL [-d TARGET_DEVICE]
                     [-hint {throughput,tput,cumulative_throughput,ctput,latency,none}] [-niter NUMBER_ITERATIONS] [-t TIME]
                     [-b BATCH_SIZE] [-shape SHAPE] [-data_shape DATA_SHAPE] [-layout LAYOUT] [-extensions EXTENSIONS]
                     [-c PATH_TO_CLDNN_CONFIG] [-cdir CACHE_DIR] [-lfile [LOAD_FROM_FILE]] [-api {sync,async}]
                     [-nireq NUMBER_INFER_REQUESTS] [-nstreams NUMBER_STREAMS] [-inference_only [INFERENCE_ONLY]]
                     [-infer_precision INFER_PRECISION] [-ip {bool,f16,f32,f64,i8,i16,i32,i64,u8,u16,u32,u64}]
                     [-op {bool,f16,f32,f64,i8,i16,i32,i64,u8,u16,u32,u64}] [-iop INPUT_OUTPUT_PRECISION]
                     [--mean_values [R,G,B]] [--scale_values [R,G,B]] [-nthreads NUMBER_THREADS]
                     [-pin {YES,NO,NUMA,HYBRID_AWARE}] [-latency_percentile LATENCY_PERCENTILE]
                     [-report_type {no_counters,average_counters,detailed_counters}] [-report_folder REPORT_FOLDER]
                     [-json_stats [JSON_STATS]] [-pc [PERF_COUNTS]] [-pcsort {no_sort,sort,simple_sort}] [-pcseq [PCSEQ]]
                     [-exec_graph_path EXEC_GRAPH_PATH] [-dump_config DUMP_CONFIG] [-load_config LOAD_CONFIG]

Available target devices:   CPU  GPU

```

### 2. 安装开发工具和编译工具

```python
sudo apt update
sudo apt-get install python3-dev
sudo apt install build-essential
```

### 3. 创建并激活虚拟环境

```bash
python3 -m venv pcdet
source <your_folder>/pcdet/bin/activate

/home/shawn/test/test_env
source /home/shawn/test/test_env/bin/activate

```

### 4. 安装 OpenVINO

```bash
pip install openvino-dev nncf

```

### 5. 安装 PyTorch

```bash
pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cpu>
# 或者使用以下命令
pip install torch torchvision torchaudio

```

### 6. 安装依赖包

```bash

进入项目工程路径下
cd /home/shawn/OpenPCDet
pip install -r requirements.txt

```

### 7. 数据集准备

请参考 SmallMunich Prepare Dataset  生成数据集。准备好数据集后，设置环境变量：

只需要生成后的 training/velodyne_reduced/ 文件夹下的数据集就可以

```bash
export my_dataset_path=<your_dataset_folder>/training/velodyne_reduced

#for example:
export my_dataset_path=/home/shawn/OpenPCDet/datasets/training/velodyne_reduced

```

### 8. 运行 [setup.py](http://setup.py/)

```bash

python setup.py develop

```

### 9. Demo运行示例

```bash
cd <your_folder>/tools/
python demo.py --cfg_file pointpillar.yaml --num -1 --data_path $my_dataset_path

#for example 
cd /home/shawn/OpenPCDet/tools
python demo.py --cfg_file pointpillar.yaml --num -1 --data_path $my_dataset_path

python demo.py --cfg_file pointpillar.yaml --num 100 --data_path /home/shawn/OpenPCDet/datasets/training/velodyne_reduced

```

结果如图：

![Untitled](PointPillars%20Demo%20becaf0d6c66f46568ba42c7c566d65ab/Untitled.png)

## 二  模型量化

### 1. 准备环境（步骤 1.1-1.8）

按照上面的步骤准备环境。

### 2. 修改量化脚本中的数据集路径

在量化脚本 quant.py中设置数据集路径：

```python
DATA_PATH=""

#for example
DATA_PATH="/home/shawn/OpenPCDet/datasets/training/velodyne_reduced"

```

### 3. 执行量化脚本

```bash
cd <your_folder>/tools/
python quant.py

#for example 
cd /home/shawn/OpenPCDet/tools
python quant.py

```

### 4. 保存量化后的模型

量化后的模型将保存到 tools 目录，命名为 `quantized_pfe.xml`。

## 三 其他模式测试

### 多线程测试

在infer.py里修改数据集的路径

```bash
cd <your_folder>/tools/
python infer.py

#for example 
cd /home/shawn/OpenPCDet/tools
python infer.py
```

配置信息可以在 `infer.py` 中的 `CONFIG` 部分进行修改。

# 参考部分

## NPU 驱动安装，2024.2 可见NPU

```python

参考安装文档
Linux NPU Driver v1.5.0
https://github.com/intel/linux-npu-driver/releases/tag/v1.5.0

(pcdet_env) shawn@MTL:~$ benchmark_app -h
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
usage: benchmark_app [-h [HELP]] [-i PATHS_TO_INPUT [PATHS_TO_INPUT ...]] -m PATH_TO_MODEL [-d TARGET_DEVICE] [-hint {throughput,tput,cumulative_throughput,ctput,latency,none}]
                     [-niter NUMBER_ITERATIONS] [-t TIME] [-b BATCH_SIZE] [-shape SHAPE] [-data_shape DATA_SHAPE] [-layout LAYOUT] [-extensions EXTENSIONS] [-c PATH_TO_CLDNN_CONFIG] [-cdir CACHE_DIR]
                     [-lfile [LOAD_FROM_FILE]] [-api {sync,async}] [-nireq NUMBER_INFER_REQUESTS] [-nstreams NUMBER_STREAMS] [-inference_only [INFERENCE_ONLY]] [-infer_precision INFER_PRECISION]
                     [-ip {bool,f16,f32,f64,i8,i16,i32,i64,u8,u16,u32,u64}] [-op {bool,f16,f32,f64,i8,i16,i32,i64,u8,u16,u32,u64}] [-iop INPUT_OUTPUT_PRECISION] [--mean_values [R,G,B]]
                     [--scale_values [R,G,B]] [-nthreads NUMBER_THREADS] [-pin {YES,NO,NUMA,HYBRID_AWARE}] [-latency_percentile LATENCY_PERCENTILE]
                     [-report_type {no_counters,average_counters,detailed_counters}] [-report_folder REPORT_FOLDER] [-json_stats [JSON_STATS]] [-pc [PERF_COUNTS]] [-pcsort {no_sort,sort,simple_sort}]
                     [-pcseq [PCSEQ]] [-exec_graph_path EXEC_GRAPH_PATH] [-dump_config DUMP_CONFIG] [-load_config LOAD_CONFIG]

Available target devices:   CPU  NPU
(pcdet_env) shawn@MTL:~$

```

## Python 环境

- **Python 版本**: 3.10.12
- **虚拟环境**: test_env
- **已安装的 Python 包**

```
about-time                4.2.1
alive-progress            3.1.5
attrs                     23.2.0
autograd                  1.6.2
certifi                   2024.7.4
charset-normalizer        3.3.2
cma                       3.2.2
contourpy                 1.2.1
cycler                    0.12.1
defusedxml                0.7.1
Deprecated                1.2.14
dill                      0.3.8
easydict                  1.13
filelock                  3.15.4
fonttools                 4.53.1
fsspec                    2024.6.1
future                    1.0.0
grapheme                  0.6.0
idna                      3.7
imageio                   2.34.2
Jinja2                    3.1.4
joblib                    1.4.2
jsonschema                4.23.0
jsonschema-specifications 2023.12.1
jstyleson                 0.0.2
kiwisolver                1.4.5
lazy_loader               0.4
llvmlite                  0.43.0
markdown-it-py            3.0.0
MarkupSafe                2.1.5
matplotlib                3.9.1
mdurl                     0.1.2
mpmath                    1.3.0
natsort                   8.4.0
networkx                  3.1
ninja                     1.11.1.1
nncf                      2.12.0
numba                     0.60.0
numpy                     1.26.4
nvidia-cublas-cu12        12.1.3.1
nvidia-cuda-cupti-cu12    12.1.105
nvidia-cuda-nvrtc-cu12    12.1.105
nvidia-cuda-runtime-cu12  12.1.105
nvidia-cudnn-cu12         9.1.0.70
nvidia-cufft-cu12         11.0.2.54
nvidia-curand-cu12        10.3.2.106
nvidia-cusolver-cu12      11.4.5.107
nvidia-cusparse-cu12      12.1.0.106
nvidia-nccl-cu12          2.20.5
nvidia-nvjitlink-cu12     12.5.82
nvidia-nvtx-cu12          12.1.105
opencv-python             4.10.0.84
openvino                  2024.3.0
openvino-dev              2024.3.0
openvino-telemetry        2024.1.0
packaging                 24.1
pandas                    2.2.2
pcdet                     0.3.0+6e282ba /home/shawn/OpenPCDet
pillow                    10.4.0
pip                       22.0.2
protobuf                  5.27.3
psutil                    6.0.0
pydot                     2.0.0
Pygments                  2.18.0
pymoo                     0.6.1.3
pyparsing                 3.1.2
python-dateutil           2.9.0.post0
pytz                      2024.1
PyYAML                    6.0.1
referencing               0.35.1
requests                  2.32.3
rich                      13.7.1
rpds-py                   0.19.1
scikit-image              0.22.0
scikit-learn              1.5.1
scipy                     1.13.1
setuptools                59.6.0
six                       1.16.0
sympy                     1.13.1
tabulate                  0.9.0
tensorboardX              2.6.2.2
threadpoolctl             3.5.0
tifffile                  2024.7.24
torch                     2.4.0
torchaudio                2.4.0
torchvision               0.19.0
tqdm                      4.66.4
triton                    3.0.0
typing_extensions         4.12.2
tzdata                    2024.1
urllib3                   2.2.2
wrapt                     1.16.0

```