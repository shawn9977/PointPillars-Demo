import os.path
import queue
import threading
import time
from pathlib import Path

from pcdet.models import build_network
from pcdet.models.model_utils import model_nms_utils

from pcdet.utils import common_utils

import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infer.utils import get_cfg, load_data_to_gpu
from infer.datasets import DemoDataset
from infer.pfe import Pfe
from infer.rpn import Rpn

CONFIG = {
    "pfe_model": {
        "model_path": "pfe.xml",
        "device_name": "GPU",
    },
    "rpn_model": {
        "model_path": "rpn.xml",
        "device_name": "GPU",
    },
    "data_path": "/home/shawn/OpenPCDet/datasets/training/velodyne_reduced",
    "cfg_file": "pointpillar.yaml",
    "args": {
        "ram": False,
        "data_num": 200,
        "ext": ".bin"
    }
}


class Demo:
    def __init__(self):
        # load model
        self.pfe = Pfe(CONFIG['pfe_model'])
        self.rpn = Rpn(CONFIG['rpn_model'])

        # data queue
        self.pfe_res_queue = queue.Queue(maxsize=30)
        self.scatter_res_queue = queue.Queue(maxsize=30)
        # self.rpn_res_queue = SelfQueue(0)
        self.rpn_res_queue = queue.Queue(maxsize=30)
        self.bbox_res_queue = queue.Queue(maxsize=30)
        # self.finall_queue = queue.Queue()

        # load datasets
        logger = common_utils.create_logger()
        cfg = get_cfg(CONFIG['cfg_file'])
        self.model_cfg = cfg.MODEL
        args = {
            "dataset_cfg": cfg.DATA_CONFIG,
            "class_names": cfg.CLASS_NAMES,
            "training": False,
            "root_path": Path(CONFIG['data_path']),
            "ram": CONFIG['args']['ram'],
            "ext": CONFIG['args']['ext'],
            "logger": logger,
            "data_num": CONFIG['args']['data_num'],
        }
        self.datasets = DemoDataset(**args)
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.datasets)
        device = 'cpu'
        self.model.to(device)
        self.model.eval()
        self.start_time = None

    def data_handler(self, index, data):
        data = self.datasets.collate_batch([data])
        data['frameid'] = index
        load_data_to_gpu(data)
        return data

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        frameid = batch_dict['frameid']
        pred_dicts = []
        # start_time = time.perf_counter()
        a_mask = batch_dict['anchor_mask']
        batch_cls_preds = batch_dict['batch_cls_preds'].squeeze(0)
        batch_box_preds = batch_dict['batch_box_preds'].squeeze(0)

        # print(a_mask.shape)
        # print(batch_box_preds.shape)
        # print(batch_cls_preds.shape)

        box_preds = batch_box_preds[a_mask]
        cls_preds = batch_cls_preds[a_mask]

        total_scores = torch.sigmoid(cls_preds)
        nms_score_threshold = post_process_cfg.SCORE_THRESH  # 0.05
        top_scores = total_scores.squeeze(-1)
        thresh = torch.tensor(
            [nms_score_threshold],
            device='cpu').type_as(total_scores)

        top_scores_keep = (top_scores >= thresh)
        # top_scores = top_scores.masked_select(top_scores_keep)

        box_preds = box_preds[top_scores_keep]
        cls_preds = top_scores[top_scores_keep]

        # cls_preds, label_preds = torch.max(cls_preds, dim=-1)
        label_preds = torch.ones((len(cls_preds),), dtype=torch.int64)  # hard coding, as there is only one type

        selected, selected_scores = model_nms_utils.class_agnostic_nms(
            box_scores=cls_preds, box_preds=box_preds,
            nms_config=post_process_cfg.NMS_CONFIG)

        final_scores = selected_scores
        final_labels = label_preds[selected]
        final_boxes = box_preds[selected]
        record_dict = {
            'pred_boxes': final_boxes,
            'pred_scores': final_scores,
            'pred_labels': final_labels,
            'pred_frameid': frameid
        }
        pred_dicts.append(record_dict)
        # print('post processing: %.2fms, detect %.f objects' %((time.perf_counter() - start_time)*1000, len(final_labels)))

        return pred_dicts

    def model_pfe_infer(self):
        print("model_pfe_infer running...")
        self.start_time = time.perf_counter()
        # pfe infer
        with torch.no_grad():
            for index, dataset in enumerate(self.datasets):
                dataset = self.data_handler(index, dataset)
                self.pfe.async_infer(index, dataset)
                res = self.pfe.wait_res()
                self.pfe_res_queue.put(res)
            self.pfe_res_queue.put(None)

    def model_scatter_infer(self):
        print("model_scatter_infer running...")
        with torch.no_grad():
            while True:
                item = self.pfe_res_queue.get()
                if item is None:
                    self.scatter_res_queue.put(None)
                    return
                data = self.model.scatter(item)
                self.scatter_res_queue.put(data)

    def model_rpn_infer(self):
        print("model_rpn_infer running...")
        with torch.no_grad():
            while True:
                item = self.scatter_res_queue.get()
                if item is None:
                    self.rpn_res_queue.put(None)
                    return
                self.rpn.async_infer(item)
                res = self.rpn.wait_res()

                self.rpn_res_queue.put(res)

    def model_bbox_infer(self):
        print("model_bbox_infer running...")
        with torch.no_grad():
            while True:
                item = self.rpn_res_queue.get()
                if item is None:
                    self.bbox_res_queue.put(None)
                    return
                data = self.model.bbox(item)
                self.bbox_res_queue.put(data)

    def post_processing_infer(self):
        print("post_processing running...")
        n = 0
        with torch.no_grad():
            while True:
                item = self.bbox_res_queue.get()
                if item is None:
                    # self.finall_queue.put(None)
                    return
                data = self.post_processing(item)
                # self.finall_queue.put(data)
                n += 1
                if n % 50 == 0:
                    print("Processing Index", f"{n}/{len(self.datasets)}")

    def full_infer(self):
        """
        开启五个线程，分别处理五个步骤
        :return:
        """
        pfe_thread = threading.Thread(target=self.model_pfe_infer)
        scatter = threading.Thread(target=self.model_scatter_infer)
        rpn_thread = threading.Thread(target=self.model_rpn_infer)
        bbox_thread = threading.Thread(target=self.model_bbox_infer)
        post_processing = threading.Thread(target=self.post_processing_infer)

        pfe_thread.start()
        scatter.start()
        rpn_thread.start()
        bbox_thread.start()
        post_processing.start()

        pfe_thread.join()
        scatter.join()
        rpn_thread.join()
        bbox_thread.join()
        post_processing.join()

        print(len(self.datasets) / (time.perf_counter() - self.start_time))


if __name__ == "__main__":
    demo = Demo()
    demo.full_infer()
