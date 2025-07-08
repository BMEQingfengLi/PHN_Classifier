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
import copy
from torch.autograd import Variable
from Utils.PFEN import PFEN_savefeature as PFEN_savefeature
from Utils.CropPatchAcrossCenter import croppatchacrosscenter
from Utils.ZeroBubble import ZeroBubble

def ConfidenceMapAndTop10FeatureGenerator(model_pth,
                                          imgrootpth,
                                          saveattention_imgname,
                                          val_namelist_np,
                                          patchsize,
                                          imgname,
                                          save_top50_featurevectornp_pth):
    '''

    :param model_pth:
    :param imgrootpth:
    :param saveattention_imgname:
    :param val_namelist_np:
    :param patchsize:
    :param imgname:
    :param save_top10_featurevectornp_pth:
    :return:
    '''
    start_time = time.time()

    # define model
    model = PFEN(out_channel=2)
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()
    param = torch.load(model_pth)
    model.load_state_dict(param)

    # define model
    model_savefeature = PFEN_savefeature(out_channel=2)
    model_savefeature.cuda()
    model_savefeature = nn.DataParallel(model_savefeature)
    model_savefeature.eval()
    param = torch.load(model_pth)
    model_savefeature.load_state_dict(param)

    img_counter = 0
    for idx in range(len(val_namelist_np)):
        img_counter += 1
        subj_name = val_namelist_np[idx]
        subj_dir = os.path.join(imgrootpth, subj_name)
        subj_imgpth = os.path.join(subj_dir, imgname)
        img = sitk.ReadImage(subj_imgpth)
        origin = img.GetOrigin()
        direction = img.GetDirection()
        spacing = img.GetSpacing()

        img_np = sitk.GetArrayFromImage(img)
        mask = copy.deepcopy(img_np)
        mask[mask > 0] = 1

        # normalization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        foreground_corrdinate_group_np = np.argwhere(img_np > 0)

        with torch.no_grad():
            # slide the box through the brain, and record the overlap time of each voxel
            radius = int(patchsize / 2)
            randomcenter_idx_list = []

            probmap_np = np.zeros(img_np.shape)

            # record patch overlap condition by a zero-value array
            overlap_record_np = np.zeros(img_np.shape)

            for counter in range(300):
                randomcenter_idx_list.append(np.random.randint(len(foreground_corrdinate_group_np)))
            for randomcenter_idx in randomcenter_idx_list:
                patch_run_starttime = time.time()
                patch_center = foreground_corrdinate_group_np[randomcenter_idx]
                patch_np, refined_patch_center = croppatchacrosscenter(patch_center, img_np, radius)

                # record overlap condition
                overlap_record_np[(refined_patch_center[0] - radius):(refined_patch_center[0] + radius + 1),
                (refined_patch_center[1] - radius):(refined_patch_center[1] + radius + 1),
                (refined_patch_center[2] - radius):(refined_patch_center[2] + radius + 1)] += 1
                patch_np = patch_np[np.newaxis, np.newaxis, :, :, :]
                patch_tensor = torch.from_numpy(patch_np).float().cuda()
                label_pred_prob = model(patch_tensor)

                # record probability
                probmap_np[(refined_patch_center[0] - radius):(refined_patch_center[0] + radius + 1),
                (refined_patch_center[1] - radius):(refined_patch_center[1] + radius + 1),
                (refined_patch_center[2] - radius):(refined_patch_center[2] + radius + 1)] += float(
                    label_pred_prob[0][1])

            # fill 0 in overlap_record_np for further division
            for zero_coordinate in np.argwhere(overlap_record_np == 0):
                overlap_record_np[zero_coordinate[0], zero_coordinate[1], zero_coordinate[2]] = 1

            probmap_np_tmp = probmap_np / overlap_record_np
            inf_coodrinate_group = np.argwhere(np.isinf(probmap_np_tmp))
            for inf_coordinate in inf_coodrinate_group:
                probmap_np_tmp[inf_coordinate[0],
                               inf_coordinate[1],
                               inf_coordinate[2]] = 0

            # cofidence normalization
            probmap_np_tmp = 0.5 + abs(probmap_np_tmp - 0.5)

            # mask background
            probmap_np_tmp = probmap_np_tmp * mask
            probmap_np = probmap_np_tmp

            # fill 0 in probmap_np=1 for further division
            for one_coordinate in np.argwhere(probmap_np == 1):
                probmap_np[one_coordinate[0], one_coordinate[1], one_coordinate[2]] = 0

            probmap_img = sitk.GetImageFromArray(probmap_np)
            probmap_img.SetOrigin(origin)
            probmap_img.SetDirection(direction)
            probmap_img.SetSpacing(spacing)
            sitk.WriteImage(probmap_img, os.path.join(subj_dir, saveattention_imgname))

            # Get top 10 feature vector
            top_featurenp_list = []
            patch_amount = 10
            for patchidx in range(patch_amount):
                currentpatch_center = np.argwhere(probmap_np == probmap_np.max())[0]
                currentpatch_np, refined_center_coordinate_np = croppatchacrosscenter(center_coordinate_np=currentpatch_center,
                                                        origin_img_np=img_np,
                                                        patch_radius=int(patchsize / 2))
                # calculate bubbled attention map
                probmap_np = ZeroBubble(refined_center_coordinate_np, probmap_np, int(patchsize / 2))
                currentpatch_np = currentpatch_np[np.newaxis, np.newaxis, :, :, :]

                # Extract feature from current patch
                currentpatch_tensor = torch.from_numpy(currentpatch_np).float()
                currentpatch_tensor = Variable(currentpatch_tensor).cuda()
                currentpatch_extractedfeature = model_savefeature(currentpatch_tensor)
                currentpatch_extractedfeature_aslist = currentpatch_extractedfeature[0].tolist()
                top_featurenp_list.append(currentpatch_extractedfeature_aslist)
            np.save(os.path.join(subj_dir, save_top10_featurevectornp_pth), np.array(top_featurenp_list))

            img_counter += 1
            print('Total time cost: %.2fs' % (time.time() - start_time))
            print('%d/%d is done' % (img_counter, len(val_namelist_np)))

