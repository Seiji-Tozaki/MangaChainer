#!/usr/bin/env python

import numpy as np
import chainer
import six
import os
import sys
import glob

from chainer import cuda, optimizers, serializers, Variable
import cv2

class Image2ImageDataset(chainer.dataset.DatasetMixin):

    def __init__(self, datFilePath, root1='./input', root2='./terget', dtype=np.float32, leak=(0, 0), root_ref = None, train=False):
        if isinstance(datFilePath, six.string_types):
            with open(datFilePath) as paths_file:
                datFilePath = [path.strip() for path in paths_file]
        self._datFilePath = datFilePath
        self._root1 = root1
        self._root2 = root2
        self._root_ref = root_ref
        self._dtype = dtype
        self._leak = leak
        self._img_dict = {}
        self._train = train

    def __len__(self):
        return len(self._datFilePath)

    #!!!この関数がイテレータから呼び出されている!!!
    def get_example(self, i, minimize=False, log=False, bin_r=0):
        if self._train:
            bin_r = 0.9
        
        path1 = os.path.join(self._root1 , self._datFilePath[i])
        path2 = os.path.join(self._root2 , self._datFilePath[i])

        #グレースケールで読み込み　サイズ（高さ x 幅）
        image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        image1 = np.asarray(image1, self._dtype)
        image2 = np.asarray(image2, self._dtype)

        #128 * 128　にリサイズ, 値を0~1の間に変換する
        image1 = cv2.resize(image1, (128, 128)) / 255.0
        image2 = cv2.resize(image2, (128, 128)) / 255.0

        return image1, image2