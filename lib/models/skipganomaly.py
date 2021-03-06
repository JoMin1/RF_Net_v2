"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lib.models.networks import NetD, weights_init, define_G, define_D, get_scheduler
from lib.visualizer import Visualizer
from lib.loss import l2_loss
from lib.evaluate import roc
from lib.models.basemodel import BaseModel

import cv2


class Skipganomaly(BaseModel):
    """GANomaly Class
    """
    @property
    def name(self): return 'skipganomaly'

    def __init__(self, opt, data=None):
        super(Skipganomaly, self).__init__(opt, data)
        ##

        # -- Misc attributes
        self.add_noise = True
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = define_G(self.opt, norm='batch', use_dropout=False, init_type='normal')
        self.netd = define_D(self.opt, norm='batch', use_sigmoid=False, init_type='normal')

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        if self.opt.verbose:
            print(self.netg)
            print(self.netd)

        ##
        # Loss Functions
        self.l_adv = nn.BCELoss()
        self.l_con = nn.L1Loss()
        self.l_lat = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.noise = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizers  = []
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def forward(self):
        self.forward_g()
        self.forward_d()

    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake = self.netg(self.input + self.noise)

    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake)

    def backward_g(self, data):
        """ Backpropagate netg
        """
        if data[1].item() == 0:
            self.err_g_con = self.l_con(self.fake, self.input)
        else:
            self.err_g_con = torch.pow(self.l_con(self.fake, self.input), -1)


        self.err_g = self.err_g_con
        self.err_g.backward(retain_graph=True)

    def backward_d(self, data):
        pred_fake, feat_fake = self.netd(self.fake.detach())

        if data[1].item() == 0:
            self.err_d_lat = self.l_lat(feat_fake, self.feat_real)
        else:
            self.err_d_lat = torch.pow(self.l_lat(feat_fake, self.feat_real), -1)

        print("   ------ ", self.err_d_lat.item(), data[1].item())

        # Combine losses.
        self.err_d = self.err_d_lat
        self.err_d.backward()

    ##
    def optimize_params(self, data):
        """ Optimize netD and netG  networks.
        """
        if data[1].item() == 0:
            self.optimizer_g.zero_grad()
            self.optimizer_d.zero_grad()

            self.forward()

            self.backward_g(data)
            self.backward_d(data)

            self.optimizer_g.step()
            self.optimizer_d.step()
        else:
            # self.optimizer_g.zero_grad()
            self.optimizer_d.zero_grad()

            self.forward()

            self.backward_d(data)

            self.optimizer_d.step()





    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            data ([type]): data for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long,    device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            self.recon_an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.recon_gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.recon_o = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.recon_i = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            self.feat_an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.feat_gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.feat_R = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32,device=self.device)
            self.feat_F = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32,device=self.device)

            self.disc_an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.disc_gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.disc_R = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32,device=self.device)
            self.disc_F = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32,device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            rec_wei = 0.9
            lat_wei = 0.1

            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)

                self.fake = self.netg(self.input)
                self.real_disc, self.real_feat = self.netd(self.input)      # netg??? ????????? ???????????? ????????? (?????? ??????)
                self.fake_disc, self.fake_feat = self.netd(self.fake)

                # print('fake_disc : {}, real_disc : {}'.format(self.fake_disc, self.real_disc))

                si = self.input.size()
                sz = self.real_feat.size()
                sd = self.real_disc.size()

                rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                feat = (self.real_feat - self.fake_feat).view(sz[0], sz[1] * sz[2] * sz[3])
                adv_real = (self.real_disc - self.real_label).view(sd[0], 1)
                adv_fake = (self.fake_disc - self.fake_label).view(sd[0], 1)

                error_recon = torch.mean(torch.pow(rec, 2), dim=1)
                error_feat = torch.mean(torch.pow(feat, 2), dim=1)
                error_discL1 = torch.mean(torch.abs(adv_real), dim=1) + torch.mean(torch.abs(adv_fake), dim=1)
                error_discL2 = torch.mean(torch.pow(adv_real, 2), dim=1) + torch.mean(torch.pow(adv_fake, 2), dim=1)
                error = rec_wei * error_recon + lat_wei * error_feat

                time_o = time.time()

                """ reconstruction """
                self.recon_an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error_recon.size(0)] = error_recon.reshape(error_recon.size(0))
                self.recon_gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error_recon.size(0)] = self.gt.reshape(error_recon.size(0))
                # self.recon_i[i * self.opt.batchsize: i * self.opt.batchsize + error_recon.size(0), :] = self.input.reshape(error_recon.size(0), self.opt.nz)
                # self.recon_o[i * self.opt.batchsize: i * self.opt.batchsize + error_recon.size(0), :] = self.fake.reshape(error_recon.size(0), self.opt.nz)

                """ feature """
                self.feat_an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error_feat.size(0)] = error_feat.reshape(error_feat.size(0))
                self.feat_gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error_feat.size(0)] = self.gt.reshape(error_feat.size(0))
                # self.feat_R[i * self.opt.batchsize: i * self.opt.batchsize + error_feat.size(0), :] = self.real_feat.reshape(error_feat.size(0), self.opt.nz)
                # self.feat_F[i * self.opt.batchsize: i * self.opt.batchsize + error_feat.size(0), :] = self.fake_feat.reshape(error_feat.size(0), self.opt.nz)

                """ AUC """
                self.an_scores[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + error.size(0)] = self.gt.reshape(error.size(0))
                # self.latent_i[i*self.opt.batchsize : i*self.opt.batchsize+error_latent.size(0), :] = latent_i.reshape(error_latent.size(0), self.opt.nz)
                # self.latent_o[i*self.opt.batchsize : i*self.opt.batchsize+error_latent.size(0), :] = latent_o.reshape(error_latent.size(0), self.opt.nz)

                """ adv L2 """
                self.disc_an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error_discL2.size(0)] = error_discL2.reshape(error_discL2.size(0))
                self.disc_gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error_discL2.size(0)] = self.gt.reshape(error_discL2.size(0))
                # self.disc_R[i * self.opt.batchsize: i * self.opt.batchsize + error_disc.size(0), :] = self.real_disc.reshape(error_disc.size(0), self.opt.nz)
                # self.disc_F[i * self.opt.batchsize: i * self.opt.batchsize + error_disc.size(0), :] = self.fake_disc.reshape(error_disc.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)


            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]

            """ reconstruction """
            self.recon_an_scores = (self.recon_an_scores - torch.min(self.recon_an_scores)) / (torch.max(self.recon_an_scores) - torch.min(self.recon_an_scores))
            print('recon : ', self.recon_an_scores)

            recon_auc = roc(self.recon_gt_labels, self.recon_an_scores)

            """ feature """
            self.feat_an_scores = (self.feat_an_scores - torch.min(self.feat_an_scores)) / (torch.max(self.feat_an_scores) - torch.min(self.feat_an_scores))
            print('feature : ', self.feat_an_scores)

            feat_auc = roc(self.feat_gt_labels, self.feat_an_scores)

            """ 0.1 : 0.9 """
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            print('AUC : ', self.an_scores)

            auc = roc(self.gt_labels, self.an_scores)

            """ Norm """
            self.Norm_an_scores = self.recon_an_scores + self.feat_an_scores
            self.Norm_an_scores = (self.Norm_an_scores - torch.min(self.Norm_an_scores)) / (torch.max(self.Norm_an_scores) - torch.min(self.Norm_an_scores))
            print('norm_AUC : ', self.Norm_an_scores)

            norm_auc = roc(self.disc_gt_labels, self.Norm_an_scores)

            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('   recon_AUC', recon_auc), ('   feat_AUC', feat_auc), ('   norm AUC', norm_auc), ('   AUC', auc)])


            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance