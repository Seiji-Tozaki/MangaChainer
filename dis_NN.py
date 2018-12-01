#!/usr/bin/env python

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers

#Discriminator
class DIS_NN(chainer.Chain):
    
    def __init__(self):
        #initialize weight
        w = chainer.initializers.Normal(scale=0.02, dtype=None)
        super(DIS_NN, self).__init__()
        #define layer
        with self.init_scope():
            self.c1=L.Convolution2D(3, 32, 4, 2, 1)
            self.c2=L.Convolution2D(32, 32, 3, 1, 1)
            self.c3=L.Convolution2D(32, 64, 4, 2, 1)
            self.c4=L.Convolution2D(64, 64, 3, 1, 1)
            self.c5=L.Convolution2D(64, 128, 4, 2, 1)
            self.c6=L.Convolution2D(128, 128, 3, 1, 1)
            self.c7=L.Convolution2D(128, 256, 4, 2, 1)
            self.l8l=L.Linear(None, 2, \
                         initialW=chainer.initializers.HeNormal( \
                             math.sqrt(0.02 * math.sqrt(8 * 8 * 256) / 2)))
            self.bnc1=L.BatchNormalization(32)
            self.bnc2=L.BatchNormalization(32)
            self.bnc3=L.BatchNormalization(64)
            self.bnc4=L.BatchNormalization(64)
            self.bnc5=L.BatchNormalization(128)
            self.bnc6=L.BatchNormalization(128)
            self.bnc7=L.BatchNormalization(256)

    def __call__(self, x):
        #define Neural Network
        h = F.relu(self.bnc1(self.c1(x)))
        h = F.relu(self.bnc2(self.c2(h)))
        h = F.relu(self.bnc3(self.c3(h)))
        h = F.relu(self.bnc4(self.c4(h)))
        h = F.relu(self.bnc5(self.c5(h)))
        h = F.relu(self.bnc6(self.c6(h)))
        h = F.relu(self.bnc7(self.c7(h)))
        return self.l8l(h)

