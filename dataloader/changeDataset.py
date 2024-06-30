import os
from pickletools import uint8
import cv2
import torch
import numpy as np

import torch.utils.data as data


class ChangeDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None):
        super(ChangeDataset, self).__init__()
        self._split_name = split_name
        self._A_format = setting['A_format']
        self._B_format = setting['B_format']
        self._gt_format = setting['gt_format']
        self._root_path = setting['root']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self.preprocess = preprocess

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
        item_name = self._file_names[index]
        A_path = os.path.join(self._root_path, 'A', item_name + self._A_format)
        B_path = os.path.join(self._root_path, 'B', item_name + self._B_format)
        gt_path = os.path.join(self._root_path, 'gt', item_name + self._gt_format)

        # Check the following settings if necessary
        A = self._open_image(A_path, cv2.COLOR_BGR2RGB)
        B = self._open_image(B_path, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)

        if self.preprocess is not None:
            A, B, gt = self.preprocess(A, B, gt)

        if self._split_name == 'train':
            A = torch.from_numpy(np.ascontiguousarray(A)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            B = torch.from_numpy(np.ascontiguousarray(B)).float()

        output_dict = dict(A=A, B=B, gt=gt, fn=str(item_name), n=len(self._file_names))

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        source = os.path.join(self._root_path, split_name+'.txt')

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            if file_name[-4]=='.':
                file_name = file_name[:-4]
            file_names.append(file_name)

        return file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors


if __name__=='__main__':
    data_setting = {'root': '/mnt/store/jparanj1/DSIFN-CD-256',
                    'A_format': '.png',
                    'B_format': '.png',
                    'gt_format': '.png',
                    'class_names': ['Background', 'Change']}
    
    dataset = ChangeDataset(data_setting, 'val')
    item = dataset[15]

    print('A_img shape: ', item['A'].shape, ' B_img shape: ', item['B'].shape, 'gt shape: ', item['gt'].shape, 'filename: ', item['fn'], ' len dataset: ', item['n'])
    # cv2.imwrite('tmpA.png', item['A'].cpu().numpy())
    # cv2.imwrite('tmpB.png', item['B'].cpu().numpy())
    # cv2.imwrite('tmp_gt.png', item['gt'].cpu().numpy())
