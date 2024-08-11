import numpy as np
import torch
import torch.nn as nn
import threading
from pathlib import Path
from openvino.runtime import Core



# BaseBEVBackbone_ASYNC=False
BaseBEVBackbone_ASYNC = True


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, openvino_ie):
        super().__init__()

        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        self.frame_id = 0
        self.event = threading.Event()
        self.queue = []
        self.request = None

        # self.ie = openvino_ie
        model_file_rpn = str(Path(__file__).resolve().parents[3] / 'tools' / 'quantized_rpn.xml')

        core = Core()
        self.net_rpn = core.read_model(model=model_file_rpn)
        self.exec_net_rpn = core.compile_model(model=self.net_rpn, device_name="GPU")

    def forward_backbone2d(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

    def callback(self, userdata):
        request, request_id, data_dict = userdata

        for k, v in request.results.items():
            name = list(k.names)[0]
            #print(f"Key: {name}, Value: {v}")
            #if name == "184":
            if name == "251":
                data_dict['batch_box_preds'] = torch.as_tensor(v)
                #print("Added batch_box_preds:", data_dict['batch_box_preds'])
            #elif name == "185":
            elif name == "252":
                data_dict['batch_cls_preds'] = torch.as_tensor(v)
                #print("Added batch_cls_preds:", data_dict['batch_cls_preds'])
            #elif name == "187":
            elif name == "254":
                data_dict['dir_cls_preds'] = torch.as_tensor(v)
                #print("Added dir_cls_preds:", data_dict['dir_cls_preds'])

        # 检查最终的 data_dict 是否包含所有必要的预测
        #print("Final data_dict keys:", data_dict.keys())

        self.queue.append(data_dict)
        self.event.set()

    def forward(self, data_dict):
        raise NotImplementedError

    def preprocessing(self, data_dict, **kwargs):
        # input_blob = next(iter(self.exec_net_rpn.input_info))
        input_blob = next(iter(self.exec_net_rpn.inputs))
        return {input_blob: data_dict['spatial_features']}


    def sync_call(self, data_dict):
        # start_time = time.perf_counter()
        inputs_param = self.preprocessing(data_dict)
        #print("inputs_param:", inputs_param)

        res = self.exec_net_rpn.infer_new_request(inputs=inputs_param)
        for k, v in res.items():
            name = list(k.names)[0]
            if name == "251":
                data_dict['batch_box_preds'] = torch.as_tensor(v)
            elif name == "252":
                data_dict['batch_cls_preds'] = torch.as_tensor(v)
            elif name == "254":
                data_dict['dir_cls_preds'] = torch.as_tensor(v)
        return data_dict

    def postprocessing(self):
        self.event.wait()
        return self.queue.pop(0)

    def async_call(self, batch_dict, inputs_param):
        self.frame_id = self.frame_id + 1
        if not self.request:
            self.request = self.exec_net_rpn.create_infer_request()
        self.request.set_callback(callback=self.callback,
                             userdata=(self.request, self.frame_id, batch_dict))
        self.event.clear()
        self.request.start_async(inputs=inputs_param)
        return
