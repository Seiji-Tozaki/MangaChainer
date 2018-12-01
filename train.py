#!/usr/bin/env python

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, training, datasets, iterators
from chainer.training import extensions

import six
import os

import argparse

import dis_NN
import gen_NN
from DCGANUpdater import DCGANUpdater
from img2imgDataset import Image2ImageDataset

chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
os.environ["CHAINER_TYPE_CHECK"] = "0"

def main():
    print('main call')
    parser = argparse.ArgumentParser(
        description='chainer line drawing COMICollorization!')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=3000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='./images/',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    root = args.dataset
    batch_size = args.batchsize
    use_device = args.gpu

    #create Neural Network
    model_gen = gen_NN.GEN_NN()
    model_dis = dis_NN.DIS_NN()

    if use_device >= 0:
        chainer.cuda.get_device_from_id(0).use()
        chainer.cuda.check_cuda_available()

        model_gen.to_gpu()
        model_dis.to_gpu()

    #データセットを作成する
    #{学習データ, 教師データ}のリスト
    dataset = Image2ImageDataset(
        "dat/images_color_train.dat", root + "line/", root + "color/", train=True)
    #繰り返し条件を作成する
    train_iter = iterators.SerialIterator(dataset, batch_size, shuffle=True)

    #Optimizerの作成
    optimizer_gen = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_gen.setup(model_gen)
    optimizer_dis = optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer_dis.setup(model_dis)

    #Trainerを作成する
    updater = DCGANUpdater(train_iter, \
        {'opt_gen':optimizer_gen, 'opt_dis':optimizer_dis}, \
        device=use_device
        )

    #機会学習の実行
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out="result")

    #学習の進展を表示する
    trainer.extend(extensions.ProgressBar(update_interval=args.display_interval))

    #中間結果を保存する
    n_save = 0
    @chainer.training.make_extension(trigger=(args.snapshot_interval, 'epoch'))
    def save_model(trainer):
        #NNのデータを保存
        global n_save
        n_save = n_save + 1
        chainer.serializers.save_hdf5('gen-'+str(n_save)+'hdf5', model_gen)
        chainer.serializers.save_hdf5('dis-'+str(n_save)+'hdf5', model_dis)
    trainer.extend(save_model)

    #ログに中間結果を保存する
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))

    #学習状況を表示する
    trainer.extend(extensions.LogReport())

    #機会学習を実行する
    trainer.run()

    #学習結果を保存する
    chainer.serializers.save_hdf5('final.hdf5', model_gen)

if __name__ == "__main__":
    main()
