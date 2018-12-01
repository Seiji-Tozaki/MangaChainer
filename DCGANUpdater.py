#!/usr/bin/env python

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable, training, datasets

#Updater
class DCGANUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(DCGANUpdater, self).__init__(
            train_iter,
            optimizer,
            device=device
        )

    #loss function for Discriminator
    #soft plusmax
    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        return loss
        
    #loss function for Generator
    #soft plusmax
    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        return loss

    def update_core(self):
        #get batchsize by each iterator
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        #get optimizer
        optimizer_gen = self.get_optimizer('opt_gen')
        optimizer_dis = self.get_optimizer('opt_dis')
        #get Neural Network model
        gen = optimizer_gen.target
        dis = optimizer_dis.target


        #get learning data
        xp = gen.xp

        target_data = xp.zeros((batchsize,3 , 128, 128)).astype("f")
        teaching_data = xp.zeros((batchsize,3 , 128, 128)).astype("f")

        for i in range(batchsize):
            target_data[i, :] = xp.asarray(batch[i][0])
            teaching_data[i, :] = xp.asarray(batch[i][1])
        target_data = Variable(target_data)
        teaching_data = Variable(teaching_data)

        #generate fake image to be discriminated from learning data
        x_fake = gen(target_data)
        #discriminate fake image
        y_fake = dis(x_fake)
        #discriminate teacher image
        y_real = dis(teaching_data)

        #learn Neural Network
        optimizer_dis.update(self.loss_dis, dis, y_fake, y_real)
        optimizer_gen.update(self.loss_gen, gen, y_fake)

