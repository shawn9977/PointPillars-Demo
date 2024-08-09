from .detector3d_template import Detector3DTemplate
import time

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.end_time = []
        self.pfe = self.module_list[0] #PillarVFE
        self.scatter = self.module_list[1] #PointPillarScatter
        self.rpn = self.module_list[2] #BaseBEVBackbone
        self.bbox = self.module_list[3] #AnchorHeadSingle

    def balance(self, batch_dict):
        if batch_dict is None:              ## None means this is the last inference
            #print("balance: batch_dict is None, performing postprocessing.")
            batch_dict_pre = self.rpn.postprocessing()
            batch_dict_pre = self.bbox(batch_dict_pre)
            pred_dicts = self.post_processing(batch_dict_pre)
            self.end_time.append(time.perf_counter())
            return pred_dicts
        #print("balance: initial batch_dict keys:", batch_dict.keys())
        if (batch_dict['frameid'] == 0):
            self.end_time = []
            batch_dict = self.pfe.sync_call(batch_dict)
            #print("balance: batch_dict keys after sync_call:", batch_dict.keys())

            batch_dict = self.scatter(batch_dict)
            #print("balance: batch_dict keys after scatter:", batch_dict.keys())

            inputs = self.rpn.preprocessing(batch_dict)
            #print("balance: inputs keys after rpn.preprocessing:", inputs.keys())
            self.rpn.async_call(batch_dict, inputs)
            #print("balance: after rpn.async_call")

            return None
        inputs = self.pfe.preprocessing(batch_dict)
        #print("balance: inputs keys after pfe.preprocessing:", inputs.keys())
    
        self.pfe.async_call(batch_dict, inputs)
        #print("balance: after pfe.async_call")

        batch_dict_pre = self.rpn.postprocessing()
        #print("balance: batch_dict_pre keys after rpn.postprocessing:", batch_dict_pre.keys())

        batch_dict_pre = self.bbox(batch_dict_pre)
        #print("balance: batch_dict_pre keys after bbox:", batch_dict_pre.keys())

        pred_dicts = self.post_processing(batch_dict_pre)
        #print("balance: batch_dict keys after pfe.postprocessing:", batch_dict.keys())

        self.end_time.append(time.perf_counter())
        batch_dict = self.pfe.postprocessing()
        batch_dict = self.scatter(batch_dict)
        #print("balance: batch_dict keys after second scatter:", batch_dict.keys())

        inputs = self.rpn.preprocessing(batch_dict)
        #print("balance: inputs keys after second rpn.preprocessing:", inputs.keys())

        self.rpn.async_call(batch_dict, inputs)
        #print("balance: after second rpn.async_call")

        return pred_dicts

    def throughput(self, batch_dict):
        if batch_dict is None:              ## None means this is the last inference
            pred_dicts = []
            batch_dict_pre = self.rpn.postprocessing()
            batch_dict_pre = self.bbox(batch_dict_pre)
            pred_dicts.append(self.post_processing(batch_dict_pre))
            self.end_time.append(time.perf_counter())

            batch_dict_pre = self.pfe.postprocessing()
            batch_dict_pre = self.scatter(batch_dict_pre)
            batch_dict_pre = self.rpn.sync_call(batch_dict_pre)
            batch_dict_pre = self.bbox(batch_dict_pre)
            pred_dicts.append(self.post_processing(batch_dict_pre))
            self.end_time.append(time.perf_counter())
            return pred_dicts

        if (batch_dict['frameid'] == 0):
            self.end_time = []
            inputs = self.pfe.preprocessing(batch_dict)
            self.pfe.async_call(batch_dict, inputs)
            return None
        if (batch_dict['frameid'] == 1):
            inputs = self.pfe.preprocessing(batch_dict)
            batch_dict_pre = self.pfe.postprocessing()
            self.pfe.async_call(batch_dict, inputs)

            batch_dict_pre = self.scatter(batch_dict_pre)
            inputs = self.rpn.preprocessing(batch_dict_pre)
            self.rpn.async_call(batch_dict_pre, inputs)

            return None

        inputs = self.pfe.preprocessing(batch_dict)
        batch_dict_pre = self.pfe.postprocessing()
        self.pfe.async_call(batch_dict, inputs)

        batch_dict_pre = self.scatter(batch_dict_pre)
        inputs = self.rpn.preprocessing(batch_dict_pre)
        batch_dict_pre2 = self.rpn.postprocessing()
        self.rpn.async_call(batch_dict_pre, inputs)

        batch_dict_pre2 = self.bbox(batch_dict_pre2)
        pred_dicts = self.post_processing(batch_dict_pre2)
        self.end_time.append(time.perf_counter())

        return pred_dicts

    def latency(self, batch_dict):
        if batch_dict is None:              ## None means this is the last inference
            return None
        if (batch_dict['frameid'] == 0):
            self.end_time = []
        batch_dict = self.pfe.sync_call(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.rpn.sync_call(batch_dict)
        batch_dict = self.bbox(batch_dict)
        pred_dicts = self.post_processing(batch_dict)
        self.end_time.append(time.perf_counter())
        return pred_dicts


    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
