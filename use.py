#!/usr/bin/env python

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, training, datasets, Variable
from chainer.training import extensions
import cv2

import six
import os
import glob
import codecs

import argparse

import dis_NN
import gen_NN
import DCGANUpdater
import img2imgDataset

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

def main():
    parser = argparse.ArgumentParser(
        description='chainer line drawing COMICollorization!')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--input', '-i', default='./use_input/',
                        help='Directory of image files.')
    args = parser.parse_args()
    
    use_device = args.gpu
    root = args.input

    #create Neural Network
    model_gen = gen_NN.GEN_NN()

    if use_device >= 0:
        chainer.cuda.get_device_from_id(0).use()
        chainer.cuda.check_cuda_available()
        model_gen.to_gpu()

    #load model
    chainer.serializers.load_hdf5('final.hdf5', model_gen)

    #get XP
    xp = model_gen.xp

    #データセットを作成する
    #学習データのリスト
    files = glob.glob(root+ "/*.jpg")
    batchsize = len(files)
    images = []
    for file in files:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = np.asarray(image, np.float32)
        image = cv2.resize(image, (128, 128)) / 255.0
        images.append(image)
        del image

    target_data = xp.zeros((batchsize, 3, 128, 128)).astype("f")

    for i in range(batchsize):
        target_data[i] = xp.asarray(images[i])

    target_data = Variable(target_data)

    with chainer.using_config('train', False):
        gen_data = model_gen(target_data)
    
    f = codecs.open('vectors.txt', 'w', 'utf8')
    for i in range(batchsize):
        result_image = xp.zeros((128, 128), dtype=np.uint8)
        dst = gen_data.data[i] * 255.0

        if use_device >= 0:
            dst = chainer.cuda.to_cpu(dst)
        
        #(c, h, w)から(h, w, c)への変換
        result_image = cv2.cvtColor(dst.transpose(1,2,0), cv2.COLOR_RGB2GRAY)
        cv2.imwrite('use_output/gen-' + str(i) + '.jpg', result_image)

        #画像の元となったベクトルを保存する
        f.write(','.join([str(j) for j in gen_data.data[i][:][:]]))
        f.write('¥n')
    f.close()

if __name__ == "__main__":
    main()