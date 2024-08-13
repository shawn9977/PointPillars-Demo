import glob
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import yaml
from easydict import EasyDict
from pcdet.utils import common_utils

from pcdet.datasets import DatasetTemplate
import nncf

from pcdet.models.detectors import PointPillar

CFG_FILE = "pointpillar.yaml"
DATA_PATH = "/home/shawn/OpenPCDet/datasets/training/velodyne_reduced"
PFE_MODEL = "pfe.xml"

RPN_MODEL = "rpn.xml"
RPN_QUANT_MODEL = "quantized_rpn.xml"


def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
        merge_new_config(config=config, new_config=new_config)

    return config


def get_cfg(cfg_file):
    cfg = EasyDict()
    cfg.LOCAL_RANK = 0
    cfg_from_yaml_file(cfg_file, cfg)
    return cfg


class DemoDataset(DatasetTemplate, torch.utils.data.DataLoader):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', ram=False):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        DatasetTemplate.__init__(
            self, dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        assert data_file_list, 'Make sure there are point data (.bin or .npy) in the folder.'

        data_file_list.sort()
        self.sample_file_list = data_file_list

        self.points_set = []
        if ram:
            for idx in range(len(self.sample_file_list)):
                if self.ext == '.bin':
                    points = np.fromfile(self.sample_file_list[idx], dtype=np.float32).reshape(-1, 4)
                elif self.ext == '.npy':
                    points = np.load(self.sample_file_list[idx])
                else:
                    raise NotImplementedError
                self.points_set.append(points)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.points_set:
            points = self.points_set[index]
        elif self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def __iter__(self):
        for index, item in enumerate(self.sample_file_list):
            data = self.__getitem__(index)
            data = self.collate_batch([data])
            data['frameid'] = 0
            load_data_to_gpu(data)
            #data = preprocessing(data)
            data = sync_call(data)
            data = scatter(data)
            data = rpn_preprocessing(data)

            yield data



def get_paddings_indicator(actual_num, max_num, axis=0):
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float()


def preprocessing(batch_dict):
    #print("preprocessing  :batch_dict ", batch_dict)  # 添加这行代码来打印结果中的所有键

    voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

    coors_x = coords[:, 3].float()
    coors_y = coords[:, 2].float()
    x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
    y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
    ones = torch.ones([1, 100], dtype=torch.float32, device="cpu")
    x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
    y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

    voxel_count = voxel_features.shape[1]
    mask = get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, 0).type_as(voxel_features)
    mask = torch.unsqueeze(mask, 0).type_as(voxel_features)

    pillar_x = voxel_features[:, :, 0].unsqueeze(0).unsqueeze(0)
    pillar_y = voxel_features[:, :, 1].unsqueeze(0).unsqueeze(0)
    pillar_z = voxel_features[:, :, 2].unsqueeze(0).unsqueeze(0)
    pillar_i = voxel_features[:, :, 3].unsqueeze(0).unsqueeze(0)
    num_points = voxel_num_points.float().unsqueeze(0)

    pillarx = pillar_x.numpy()
    pillary = pillar_y.numpy()
    pillarz = pillar_z.numpy()
    pillari = pillar_i.numpy()
    numpoints = num_points.numpy()
    xsub_shaped = x_sub_shaped.numpy()
    ysub_shaped = y_sub_shaped.numpy()
    mask_np = mask.numpy()

    pillar_len = pillarx.shape[2]
    if pillar_len < 12000:
        len_padding = 12000 - pillar_len
        pillarx_pad = np.pad(pillarx, ((0, 0), (0, 0), (0, len_padding), (0, 0)), 'constant', constant_values=0)
        pillary_pad = np.pad(pillary, ((0, 0), (0, 0), (0, len_padding), (0, 0)), 'constant', constant_values=0)
        pillarz_pad = np.pad(pillarz, ((0, 0), (0, 0), (0, len_padding), (0, 0)), 'constant', constant_values=0)
        pillari_pad = np.pad(pillari, ((0, 0), (0, 0), (0, len_padding), (0, 0)), 'constant', constant_values=0)
        nump_pad = np.pad(numpoints, ((0, 0), (0, len_padding)), 'constant', constant_values=0)
        xsub_pad = np.pad(xsub_shaped, ((0, 0), (0, 0), (0, len_padding), (0, 0)), 'constant', constant_values=0)
        ysub_pad = np.pad(ysub_shaped, ((0, 0), (0, 0), (0, len_padding), (0, 0)), 'constant', constant_values=0)
        mask_pad = np.pad(mask_np, ((0, 0), (0, 0), (0, len_padding), (0, 0)), 'constant', constant_values=0)
    else:
        pillarx_pad = pillarx[:, :, :12000, :]
        pillary_pad = pillary[:, :, :12000, :]
        pillarz_pad = pillarz[:, :, :12000, :]
        pillari_pad = pillari[:, :, :12000, :]
        nump_pad = numpoints[:, :12000]
        xsub_pad = xsub_shaped[:, :, :12000, :]
        ysub_pad = ysub_shaped[:, :, :12000, :]
        mask_pad = mask_np[:, :, :12000, :]

    pillar_x_tensor = torch.from_numpy(pillarx_pad)
    pillar_y_tensor = torch.from_numpy(pillary_pad)
    pillar_z_tensor = torch.from_numpy(pillarz_pad)
    pillar_i_tensor = torch.from_numpy(pillari_pad)
    num_points_tensor = torch.from_numpy(nump_pad)
    x_sub_shaped_tensor = torch.from_numpy(xsub_pad)
    y_sub_shaped_tensor = torch.from_numpy(ysub_pad)
    mask_tensor = torch.from_numpy(mask_pad)

    inputs = {'pillar_x': pillar_x_tensor,
              'pillar_y': pillar_y_tensor,
              'pillar_z': pillar_z_tensor,
              'pillar_i': pillar_i_tensor,
              'num_points_per_pillar': num_points_tensor,
              'x_sub_shaped': x_sub_shaped_tensor,
              'y_sub_shaped': y_sub_shaped_tensor,
              'mask': mask_tensor}
    return inputs


def sync_call(batch_dict):
    #print("sync_call  :batch_dict ", batch_dict)  # 添加这行代码来打印结果中的所有键

    inputs_param = preprocessing(batch_dict)

    core = ov.Core()
    net_pfe = core.read_model(PFE_MODEL)
    exec_net_pfe = core.compile_model(model=net_pfe, device_name="GPU")
    exec_net = exec_net_pfe
    res_torch = None  # 在循环前定义一个默认值
    #print("Inference results keys:inputs ", inputs_param)  # 添加这行代码来打印结果中的所有键

    res = exec_net.infer_new_request(inputs=inputs_param)
    #print("Inference results keys:", res.keys())  # 添加这行代码来打印结果中的所有键

    for k, v in res.items():
        #print(f"Key: {k}, Value shape: {v.shape}")
        if list(k.names)[0] == "173":
            res_torch = torch.as_tensor(v)
            break  # 一旦找到我们需要的，退出循环

    if res_torch is None:
        raise ValueError("Expected key '173' not found in the inference results.")
    
    #print("res_torch:", res_torch)
    voxel_features = res_torch.squeeze()
    voxel_features = voxel_features.permute(1, 0)
    batch_dict['pillar_features'] = voxel_features

    #打印 batch_dict 的键值对
    #print("sync_call  :batch_dict ", batch_dict)  # 添加这行代码来打印结果中的所有键
    return batch_dict


logger = common_utils.create_logger()
cfg = get_cfg(CFG_FILE)
args = {
    "dataset_cfg": cfg.DATA_CONFIG,
    "class_names": cfg.CLASS_NAMES,
    "training": False,
    "root_path": Path(DATA_PATH),
    "ram": False,
    "ext": ".bin",
    "logger": logger
}

# 实例化 PointPillar
pointpillar_model = PointPillar(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset = DemoDataset(**args))

# 调用实例方法
#module_list = pointpillar_model.build_networks()
#scatterfunc = module_list[1]


def scatter(batch_dict):
    #print("scatter  :batch_dict ", batch_dict)  # 添加这行代码来打印结果中的所有键

    #batch_dict = scatterfunc(batch_dict)
    batch_dict = pointpillar_model.scatter(batch_dict)

    return batch_dict


def rpn_preprocessing(batch_dict):
    #print("rpn_preprocessing  :batch_dict ", batch_dict)  # 添加这行代码来打印结果中的所有键

    core = ov.Core()
    net_rpn = core.read_model(RPN_MODEL)
    exec_net_rpn = core.compile_model(model=net_rpn, device_name="GPU")

    # input_blob = next(iter(self.exec_net_rpn.input_info))
    input_blob = next(iter(exec_net_rpn.inputs))
    return {input_blob: batch_dict['spatial_features']}



def main():
    logger = common_utils.create_logger()
    cfg = get_cfg(CFG_FILE)
    args = {
        "dataset_cfg": cfg.DATA_CONFIG,
        "class_names": cfg.CLASS_NAMES,
        "training": False,
        "root_path": Path(DATA_PATH),
        "ram": False,
        "ext": ".bin",
        "logger": logger
    }

    datasets = DemoDataset(**args)
    datasets = nncf.Dataset(datasets)

    quantize_config = {
        "fast_bias_correction": False,
        "target_device": nncf.TargetDevice.GPU,
        "subset_size": 300
    }

    core = ov.Core()
    rpn_model = core.read_model(RPN_MODEL)
    rpn_quantized_model = nncf.quantize(rpn_model, datasets)
    ov.save_model(rpn_quantized_model, RPN_QUANT_MODEL)

if __name__ == "__main__":
    main()

