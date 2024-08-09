import glob

import numpy as np
from pcdet.datasets import DatasetTemplate


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', ram=False, data_num=-1):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        # print(training)
        # input()
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        assert data_file_list, 'Make sure there are point data (.bin or .npy) in the folder.'

        data_file_list.sort()
        if data_num != -1:
            data_file_list = data_file_list[:data_num]
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
