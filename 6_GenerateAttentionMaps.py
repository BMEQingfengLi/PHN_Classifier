import pandas as pd
import torch
import random
import os
import time
import csv
import numpy as np
import SimpleITK as sitk
import torch.nn as nn

from Utils.PFEN import PFEN
from Utils.CropPatchAcrossCenter import croppatchacrosscenter
from Utils.ConfidenceMapAndFeatureGenerator import ConfidenceMapAndTop10FeatureGenerator
import copy

def GenerateAttentionMap(imgrootpth,
                         gpuidx,
                         model_pth,
                         saveattention_imgname,
                         patchsize,
                         imgname,
                         targetcsvpth,
                         label0_list,
                         label1_list,
                         save_top10_featurevectornp_pth):
    '''

    :param imgrootpth:
    :param gpuidx:
    :param model_pth:
    :param saveattention_imgname:
    :param patchsize:
    :param imgname:
    :param targetcsvpth:
    :param label0_list:
    :param label1_list:
    :param save_top10_featurevectornp_pth:
    :return:
    '''
    # set random seed
    random.seed(42)

    # set cuda
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpuidx

    origdata_valcsv = pd.read_csv(targetcsvpth)
    val_namelist_np = np.array(origdata_valcsv['name'])
    val_labellist_np = np.array(origdata_valcsv['label'])

    # target set re-label
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

    ConfidenceMapAndTop10FeatureGenerator(model_pth,
                                          imgrootpth,
                                          saveattention_imgname,
                                          val_namelist_np,
                                          patchsize,
                                          imgname,
                                          save_top10_featurevectornp_pth)


if __name__ == '__main__':
    gpuidx = 0
    patchsize = 48
    imgname = 't1_brain.nii.gz'
    savemodel_rootdir = './SelectModel_stage1'

    # HCOthers: HC as 1, Others as 0
    pair = 'HCOthers'
    label0_list = [1,2,3,4]
    label1_list = [0]
    fold = 1

    # get selected model path
    suffix = '%s_fold%d' % (pair, fold)
    selectmodel_foldpth = os.path.join(savemodel_rootdir, suffix)
    selected_model_name = os.listdir(selectmodel_foldpth)[0]
    model_pth = os.path.join(selectmodel_foldpth, selected_model_name)

    imgrootpth = './DL_trainingdata/'

    for task in ['train', 'test', 'val']:
        print('Current task: %s_fold%d  %s' % (pair, fold, task))
        targetcsvpth = './CrossValData/5foldcv_fold%d_%s.csv' % (fold, task)
        saveattention_imgname = 'attentionmap_%s_%s.nii.gz' % (suffix, task)
        save_top10_featurevectornp_pth = 'attentionmap_top10_featurevector_%s_%s.npy' % (suffix, task)
        GenerateAttentionMap(imgrootpth,
                             gpuidx,
                             model_pth,
                             saveattention_imgname,
                             patchsize,
                             imgname,
                             targetcsvpth,
                             label0_list,
                             label1_list,
                             save_top10_featurevectornp_pth)