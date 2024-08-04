import threading

import numpy as np
import openvino as ov
import torch


class Pfe:
    def __init__(self, model):
        core = ov.Core()
        self.pfe_model = core.read_model(model=model['model_path'])
        self.pfe_model = core.compile_model(model=self.pfe_model, device_name=model['device_name'])
        self.request = None
        self.event = threading.Event()
        self.queue = []
        self.frame_id = 0

    def callback(self, userdata):
        request, request_id, data_dict = userdata
        res = request.model_outputs
        for index, item in enumerate(res):
            if list(item.names)[0] == "174":
                res_torch = torch.as_tensor(request.results[index])

        voxel_features = res_torch.squeeze()
        voxel_features = voxel_features.permute(1, 0)
        data_dict['pillar_features'] = voxel_features
        self.queue.append(data_dict)
        self.event.set()

    def async_infer(self, index, batch_dict):
        # print("====== pfe async_infer:", index)
        inputs_param = self.preprocessing(batch_dict)
        self.frame_id = index
        if not self.request:
            self.request = self.pfe_model.create_infer_request()
        self.request.set_callback(callback=self.callback,
                                  userdata=(self.request, self.frame_id, batch_dict))

        self.event.clear()
        self.request.start_async(inputs=inputs_param)
        return

    def wait_res(self):
        self.event.wait()
        return self.queue.pop(0)

    def sync_call(self, batch_dict):
        inputs_param = self.preprocessing(batch_dict)
        res = self.pfe_model.infer_new_request(inputs=inputs_param)
        for k, v in res.items():
            if list(k.names)[0] == "174":
                res_torch = torch.as_tensor(v)
        voxel_features = res_torch.squeeze()
        voxel_features = voxel_features.permute(1, 0)
        batch_dict['pillar_features'] = voxel_features
        return batch_dict

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def preprocessing(self, batch_dict):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']

        coors_x = coords[:, 3].float()
        coors_y = coords[:, 2].float()
        x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
        y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
        ones = torch.ones([1, 100], dtype=torch.float32, device="cpu")
        x_sub_shaped = torch.mm(x_sub, ones).unsqueeze(0).unsqueeze(0)
        y_sub_shaped = torch.mm(y_sub, ones).unsqueeze(0).unsqueeze(0)

        voxel_count = voxel_features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
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
