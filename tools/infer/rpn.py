import threading
import openvino as ov
import torch

BaseBEVBackbone_ASYNC = True


class Rpn:
    def __init__(self, model):
        core = ov.Core()
        self.rpn_model = core.read_model(model=model['model_path'])
        self.rpn_model = core.compile_model(model=self.rpn_model, device_name=model['device_name'])
        self.request = None
        self.event = threading.Event()
        self.queue = []
        self.frame_id = 0

    def preprocessing(self, data_dict):
        input_blob = next(iter(self.rpn_model.inputs))
        return {input_blob: data_dict['spatial_features']}

    def callback(self, userdata):
        request, request_id, data_dict = userdata
        for k, v in request.results.items():
            name = list(k.names)[0]
            if name == "251":
                data_dict['batch_box_preds'] = torch.as_tensor(v)
            elif name == "252":
                data_dict['batch_cls_preds'] = torch.as_tensor(v)
            elif name == "254":
                data_dict['dir_cls_preds'] = torch.as_tensor(v)
        self.queue.append(data_dict)
        self.event.set()

    def async_infer(self, batch_dict):
        inputs_param = self.preprocessing(batch_dict)
        self.frame_id = self.frame_id + 1
        if not self.request:
            self.request = self.rpn_model.create_infer_request()
        self.request.set_callback(callback=self.callback,
                                  userdata=(self.request, self.frame_id, batch_dict))
        self.event.clear()
        self.request.start_async(inputs=inputs_param)
        return

    def wait_res(self):
        self.event.wait()
        return self.queue.pop(0)

    def sync_call(self, data_dict):
        inputs_param = self.preprocessing(data_dict)
        res = self.rpn_model.infer_new_request(inputs=inputs_param)
        for k, v in res.items():
            name = list(k.names)[0]
            if name == "251":
                data_dict['batch_box_preds'] = torch.as_tensor(v)
            elif name == "252":
                data_dict['batch_cls_preds'] = torch.as_tensor(v)
            elif name == "254":
                data_dict['dir_cls_preds'] = torch.as_tensor(v)
        return data_dict
