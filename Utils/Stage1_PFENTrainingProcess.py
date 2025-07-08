import pandas as pd
import torch
import random
import os
import time
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from torch.autograd import Variable

from Utils.PFEN import PFEN
from Utils.CreatDataset import CreatDataset
from Utils.FocalLoss import Focalloss
from Utils.MultiClassACC import MulticlassACCScore


def sen_score(true_label_list, pred_label_list_np):

    return recall_score(true_label_list, pred_label_list_np)


def spe_score(true_label_list, pred_label_list_np):
    negcase_amount = 0
    true_neg_amount = 0
    for idx in range(len(true_label_list)):
        if true_label_list[idx] == 0:
            negcase_amount += 1
            if pred_label_list_np[idx] == 0:
                true_neg_amount += 1
    spe = true_neg_amount / negcase_amount
    return spe


def training(imgrootpth,
             train_namelist_np,
             train_labellist_np,
             val_namelist_np,
             val_labellist_np,
             savepth,
             epoch,
             batchsize,
             threadnum,
             patchsize,
             imgname,
             pretrainmodelpth):
    '''

    :param imgrootpth:
    :param train_namelist_np:
    :param train_labellist_np:
    :param val_namelist_np:
    :param val_labellist_np:
    :param savepth:
    :param epoch:
    :param batchsize:
    :param threadnum:
    :param patchsize:
    :param imgname:
    :param pretrainmodelpth:
    :return:
    '''
    print('############## Training beginning... ##############')

    # temporary results collect list
    average_patch_loss_list = []
    average_patch_valloss_list = []
    average_patch_acc_list = []
    average_patch_valacc_list = []
    average_valsen_list = []
    average_valspe_list = []

    # create dataset
    train_dataset = CreatDataset(train_namelist_np,
                                 train_labellist_np,
                                 imgrootpth,
                                 patchsize,
                                 imgname,
                                 traintestflag='train')
    val_dataset = CreatDataset(val_namelist_np,
                               val_labellist_np,
                               imgrootpth,
                               patchsize,
                               imgname,
                               traintestflag='val')

    # create dataloader
    data_train_loader = DataLoader(dataset=train_dataset,
                                   batch_size=batchsize,
                                   shuffle=True,
                                   num_workers=threadnum)
    data_val_loader = DataLoader(dataset=val_dataset,
                                 batch_size=batchsize,
                                 shuffle=True,
                                 num_workers=threadnum)

    # define model
    model = PFEN(out_channel=2)
    model.cuda()
    model = nn.DataParallel(model)

    # load pre-trained model (if available)
    if os.path.isfile(pretrainmodelpth):
        model.load_state_dict(torch.load(pretrainmodelpth))

    # optimizer set
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4,
                           weight_decay=1e-8)

    # loss function set
    loss_function = Focalloss(class_num=2)

    for epoch in range(1, epoch):
        epoch_start_time = time.time()
        model.train()
        currentepoch_patch_avg_acc = 0.0
        currentepoch_patch_avg_loss = 0.0

        batch_counter = 0
        batch_start_time = time.time()
        for i, (patch_tensor, label_tensor) in enumerate(data_train_loader):
            batch_counter += 1
            label_tensor = Variable(label_tensor).cuda()
            label_tensor_reshapeaspred = label_tensor[:, 0, 0]
            patch_tensor = Variable(patch_tensor).cuda()

            # optimize according to patch
            label_pred_prob_patch = model(patch_tensor)
            label_pred_patch = label_pred_prob_patch.data.max(1)[1]
            acc_patch = MulticlassACCScore(label_tensor_reshapeaspred.cpu().detach().numpy(),
                                             label_pred_patch.cpu().detach().numpy())
            currentepoch_patch_avg_acc += acc_patch
            loss_patch = loss_function(label_pred_prob_patch, label_tensor)
            optimizer.zero_grad()
            loss_patch.backward()
            optimizer.step()
            currentepoch_patch_avg_loss += float(loss_patch)

            print('TRAINING EPOCH: %d  Batch: %d  ACC: %.4f  Loss: %.4f  time: %.2fs'
                  % (epoch, batch_counter, acc_patch, float(loss_patch), time.time() - batch_start_time))
            batch_start_time = time.time()
            torch.cuda.empty_cache()

        currentepoch_patch_avg_loss /= batch_counter
        currentepoch_patch_avg_acc /= batch_counter

        average_patch_loss_list.append(currentepoch_patch_avg_loss)
        average_patch_acc_list.append(currentepoch_patch_avg_acc)
        print('TRAINING EPOCH: %d  AvgACC: %.4f  AvgLoss: %.4f  time: %.2fs' %
              (epoch,
               currentepoch_patch_avg_acc,
               currentepoch_patch_avg_loss,
               time.time() - epoch_start_time))

        # testing
        val_start_time = time.time()
        currentepochval_patch_avg_acc = 0.0
        currentepochval_patch_avg_loss = 0.0

        batch_counter = 0
        model.eval()
        true_val_labellist = []
        pred_val_labellist = []
        for i, (patch_tensor, label_tensor) in enumerate(data_val_loader):
            batch_counter += 1
            label_tensor = Variable(label_tensor).cuda()
            label_tensor_reshapeaspred = label_tensor[:, 0, 0]
            patch_tensor = Variable(patch_tensor).cuda()

            with torch.no_grad():
                label_pred_prob_patch = model(patch_tensor)
                label_pred_patch = label_pred_prob_patch.data.max(1)[1]
                acc_patch = MulticlassACCScore(label_tensor_reshapeaspred.cpu().detach().numpy(),
                                                 label_pred_patch.cpu().detach().numpy())
                currentepochval_patch_avg_acc += acc_patch
                loss_patch = loss_function(label_pred_prob_patch, label_tensor)
                currentepochval_patch_avg_loss += float(loss_patch)

                for truelabel in label_tensor_reshapeaspred.cpu().detach().numpy():
                    true_val_labellist.append(truelabel)

                for predlabel in label_pred_patch.cpu().detach().numpy():
                    pred_val_labellist.append(predlabel)

                torch.cuda.empty_cache()

        currentepochval_patch_avg_loss /= batch_counter
        currentepochval_patch_avg_acc /= batch_counter

        currentepoch_val_sen = sen_score(true_val_labellist, pred_val_labellist)
        currentepoch_val_spe = spe_score(true_val_labellist, pred_val_labellist)

        # save model
        torch.save(model.state_dict(),
                   os.path.join(savepth, 'model_epoch_%d.pth' % epoch))
        average_patch_valloss_list.append(currentepochval_patch_avg_loss)
        average_patch_valacc_list.append(currentepochval_patch_avg_acc)

        average_valsen_list.append(currentepoch_val_sen)
        average_valspe_list.append(currentepoch_val_spe)

        print('TESTING EPOCH: %d  Avg3ACC: %.4f  Avg3SEN: %.4f  Avg3SPE: %.4f  Avg3Loss: %.4f  valtime: %.2fs' %
              (epoch,
               currentepochval_patch_avg_acc,
               currentepoch_val_sen,
               currentepoch_val_spe,
               currentepochval_patch_avg_loss,
               time.time() - val_start_time))

        # plot acc
        plt.figure(1)
        plt.xlabel('Epoch')
        plt.ylabel('ACC')
        plt.grid()
        x = range(1, epoch + 1)
        plt.plot(x, average_patch_acc_list, label='Training ACC', color='turquoise')
        plt.plot(x, average_patch_valacc_list, label='Validation ACC', color='lightseagreen')
        plt.plot(x, average_valsen_list, label='Val SEN', color='lime')
        plt.plot(x, average_valspe_list, label='Val SPE', color='green')

        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.rcParams['figure.dpi'] = 1000
        plt.rcParams['savefig.dpi'] = 1000
        plt.tight_layout()
        plt.savefig(os.path.join(savepth, 'ACC.png'))
        plt.savefig(os.path.join(savepth, 'ACC.svg'))
        np.save(os.path.join(savepth, 'average_patch_acc.npy'),
                np.array(average_patch_acc_list))
        np.save(os.path.join(savepth, 'average_patch_valacc.npy'),
                np.array(average_patch_valacc_list))

        # save temporary record
        np.save(os.path.join(savepth, 'average_val_sen.npy'),
                np.array(average_valsen_list))
        np.save(os.path.join(savepth, 'average_test_spe.npy'),
                np.array(average_valspe_list))

        plt.clf()

        # plot loss
        plt.figure(1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        x = range(1, epoch + 1)
        plt.plot(x, average_patch_loss_list, label='Training loss', color='turquoise')
        plt.plot(x, average_patch_valloss_list, label='Validation loss', color='lightseagreen')
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.rcParams['figure.dpi'] = 1000
        plt.rcParams['savefig.dpi'] = 1000
        plt.tight_layout()
        plt.savefig(os.path.join(savepth, 'Loss.png'))
        plt.savefig(os.path.join(savepth, 'Loss.svg'))
        plt.clf()
        np.save(os.path.join(savepth, 'average_patch_loss.npy'),
                np.array(average_patch_loss_list))
        np.save(os.path.join(savepth, 'average_patch_testloss.npy'),
                np.array(average_patch_valloss_list))
        plt.clf()


def Stage1Training(imgrootpth,
                   gpuidx,
                   savepth,
                   epoch,
                   batchsize,
                   threadnum,
                   patchsize,
                   imgname,
                   traincsvpth,
                   valcsvpth,
                   pretrainmodelpth,
                   label0_list,
                   label1_list):
    '''

    :param imgrootpth:
    :param gpuidx:
    :param savepth:
    :param epoch:
    :param batchsize:
    :param threadnum:
    :param patchsize:
    :param imgname:
    :param traincsvpth:
    :param valcsvpth:
    :param pretrainmodelpth:
    :param label0_list:
    :param label1_list:
    :return:
    '''
    random.seed(42)

    # set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuidx

    # load origin train and val csv
    origdata_traincsv = pd.read_csv(traincsvpth)
    origdata_valcsv = pd.read_csv(valcsvpth)
    train_namelist_np = np.array(origdata_traincsv['name'])
    train_labellist_np = np.array(origdata_traincsv['label'])
    val_namelist_np = np.array(origdata_valcsv['name'])
    val_labellist_np = np.array(origdata_valcsv['label'])

    # training set re-label
    new_trainnamelist = []
    new_trainlabellist = []
    for idx in range(len(train_namelist_np)):
        if train_labellist_np[idx] in label1_list:
            new_trainlabellist.append(1)
            new_trainnamelist.append(train_namelist_np[idx])
        if train_labellist_np[idx] in label0_list:
            new_trainlabellist.append(0)
            new_trainnamelist.append(train_namelist_np[idx])
    train_namelist_np = np.array(new_trainnamelist)
    train_labellist_np = np.array(new_trainlabellist)

    # validation set re-label
    new_valnamelist = []
    new_vallabellist = []
    for idx in range(len(val_namelist_np)):
        if val_labellist_np[idx] in label1_list:
            new_vallabellist.append(1)
            new_valnamelist.append(val_namelist_np[idx])
        if val_labellist_np[idx] in label0_list:
            new_vallabellist.append(0)
            new_valnamelist.append(val_namelist_np[idx])
    val_namelist_np = np.array(new_valnamelist)
    val_labellist_np = np.array(new_vallabellist)

    # creat savepth
    if not os.path.isdir(savepth):
        os.mkdir(savepth)

    # training
    training(imgrootpth,
             train_namelist_np,
             train_labellist_np,
             val_namelist_np,
             val_labellist_np,
             savepth,
             epoch,
             batchsize,
             threadnum,
             patchsize,
             imgname,
             pretrainmodelpth)
