#-*-coding:gbk-*-
import pandas as pd
import numpy as np

def GetParam(allinvolvedsubj_csvpth,
             subjredundant_csvpth,
             save_involved_scanparam_csvpth):
    '''

    :param allinvolvedsubj_csvpth:
    :param subjredundant_csvpth:
    :param save_involved_scanparam_csvpth:
    :return:
    '''
    allinvolvedsubj_csv = pd.read_csv(allinvolvedsubj_csvpth, encoding='gbk')
    subjgroup_inallinvolvedcsv_np = np.array(allinvolvedsubj_csv['name'])
    labelgroup_inallinvolvedcsv_np = np.array(allinvolvedsubj_csv['label'])
    researchgroup_inallinvolvedcsv_np = np.array(allinvolvedsubj_csv['dataset'])

    # load subjredundant_csv
    subjredundant_csv = pd.read_csv(subjredundant_csvpth, encoding='gbk')
    subjgroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'name'])
    repetitiontimegroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'Repetition Time1'])
    flipangelgroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'Flip Angle1'])
    echotimegroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'Echo Time1'])
    voxelsizegroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'voxel size'])
    slicenumbergroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'Slice numbers'])
    matrixsizegroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'MatrixSize1'])
    fovgroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'FoV1'])
    researchgroup_insubjredundantcsv_np = np.array(subjredundant_csv[u'researchgroup'])

    # calculate for pair 1:
    # repetition time: 2530
    # echo time: 3.65
    # flip angel: 7
    # FOV: 256*256
    # Matrix size: 256*256
    # number of slices: 224
    # slice thickness: 1
    # voxel size: 1*1*1
    pair1_hc_counter = 0
    pair1_bd_counter = 0
    pair1_mdd_counter = 0
    pair1_sz_counter = 0
    pair1_ocd_counter = 0
    print('##################### Pair1 is checking...')
    for idx_in_allinvolvedsubjcsv in range(len(subjgroup_inallinvolvedcsv_np)):
        current_subj = subjgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        current_label = labelgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        currensubj_researchgroupinallinvolvedcsv = researchgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        # if currentsubj in scanparam_csv, then record its parameters
        if current_subj in subjgroup_insubjredundantcsv_np:
            currentsubj_idx_inscanparam_csv = np.argwhere(subjgroup_insubjredundantcsv_np == current_subj)[0][0]
        # if currentsubj not in scanparam csv, then record the parameter of same research group
        else:
            if currensubj_researchgroupinallinvolvedcsv == 'WJH':
                currentsubj_idx_inscanparam_csv = np.argwhere(researchgroup_insubjredundantcsv_np == 'wangjinghong_clinicaldat')[0][0]
            else:
                for research_group_idx in range(len(researchgroup_insubjredundantcsv_np)):
                    research_group = researchgroup_insubjredundantcsv_np[research_group_idx]
                    if research_group in current_subj:
                        currentsubj_idx_inscanparam_csv = research_group_idx
                    break
        currentsubj_reptime = repetitiontimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_flipang = flipangelgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_echotime = echotimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentvoxelsize = voxelsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentslicenum = slicenumbergroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentmtxsize = matrixsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentfov = fovgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        if (currentsubj_reptime == 2530) \
            and (currentsubj_flipang == 7) \
            and (currentsubj_echotime == 3.65) \
            and (currentvoxelsize == '1*1*1') \
            and (currentslicenum == 224) \
            and (currentmtxsize == '256*256') \
            and (currentfov == '256*256'):
            if current_label == 0:
                pair1_hc_counter += 1
            elif current_label == 1:
                pair1_bd_counter += 1
            elif current_label == 2:
                pair1_mdd_counter += 1
            elif current_label == 3:
                pair1_sz_counter += 1
            elif current_label == 4:
                pair1_ocd_counter += 1
    print('HC num: %d' % pair1_hc_counter)
    print('BD num: %d' % pair1_bd_counter)
    print('MDD num: %d' % pair1_mdd_counter)
    print('SZ num: %d' % pair1_sz_counter)
    print('OCD num: %d' % pair1_ocd_counter)


    # calculate for pair 2:
    # repetition time: 2530
    # echo time: 3.65
    # flip angel: 7
    # FOV: 240*256
    # Matrix size: 240*256
    # number of slices: 224
    # slice thickness: 1
    # voxel size: 1*1*1
    pair2_hc_counter = 0
    pair2_bd_counter = 0
    pair2_mdd_counter = 0
    pair2_sz_counter = 0
    pair2_ocd_counter = 0
    print('##################### Pair2 is checking...')
    for idx_in_allinvolvedsubjcsv in range(len(subjgroup_inallinvolvedcsv_np)):
        current_subj = subjgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        current_label = labelgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        currensubj_researchgroupinallinvolvedcsv = researchgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        # if currentsubj in scanparam_csv, then record its parameters
        if current_subj in subjgroup_insubjredundantcsv_np:
            currentsubj_idx_inscanparam_csv = np.argwhere(subjgroup_insubjredundantcsv_np == current_subj)[0][0]
        # if currentsubj not in scanparam csv, then record the parameter of same research group
        else:
            if currensubj_researchgroupinallinvolvedcsv == 'WJH':
                currentsubj_idx_inscanparam_csv = np.argwhere(researchgroup_insubjredundantcsv_np == 'wangjinghong_clinicaldat')[0][0]
            else:
                for research_group_idx in range(len(researchgroup_insubjredundantcsv_np)):
                    research_group = researchgroup_insubjredundantcsv_np[research_group_idx]
                    if research_group in current_subj:
                        currentsubj_idx_inscanparam_csv = research_group_idx
                    break
        currentsubj_reptime = repetitiontimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_flipang = flipangelgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_echotime = echotimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentvoxelsize = voxelsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentslicenum = slicenumbergroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentmtxsize = matrixsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentfov = fovgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        if (currentsubj_reptime == 2530) \
            and (currentsubj_flipang == 7) \
            and (currentsubj_echotime == 3.65) \
            and (currentvoxelsize == '1*1*1') \
            and (currentslicenum == 224) \
            and (currentmtxsize == '240*256') \
            and (currentfov == '240*256'):
            if current_label == 0:
                pair2_hc_counter += 1
            elif current_label == 1:
                pair2_bd_counter += 1
            elif current_label == 2:
                pair2_mdd_counter += 1
            elif current_label == 3:
                pair2_sz_counter += 1
            elif current_label == 4:
                pair2_ocd_counter += 1
    print('HC num: %d' % pair2_hc_counter)
    print('BD num: %d' % pair2_bd_counter)
    print('MDD num: %d' % pair2_mdd_counter)
    print('SZ num: %d' % pair2_sz_counter)
    print('OCD num: %d' % pair2_ocd_counter)

    # calculate for pair 3:
    # repetition time: 2530
    # echo time: 3.65
    # flip angel: 7
    # FOV: 176*256
    # Matrix size: 176*256
    # number of slices: 224
    # slice thickness: 1
    # voxel size: 1*1*1
    pair3_hc_counter = 0
    pair3_bd_counter = 0
    pair3_mdd_counter = 0
    pair3_sz_counter = 0
    pair3_ocd_counter = 0
    print('##################### Pair3 is checking...')
    for idx_in_allinvolvedsubjcsv in range(len(subjgroup_inallinvolvedcsv_np)):
        current_subj = subjgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        current_label = labelgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        currensubj_researchgroupinallinvolvedcsv = researchgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        # if currentsubj in scanparam_csv, then record its parameters
        if current_subj in subjgroup_insubjredundantcsv_np:
            currentsubj_idx_inscanparam_csv = np.argwhere(subjgroup_insubjredundantcsv_np == current_subj)[0][0]
        # if currentsubj not in scanparam csv, then record the parameter of same research group
        else:
            if currensubj_researchgroupinallinvolvedcsv == 'WJH':
                currentsubj_idx_inscanparam_csv = np.argwhere(researchgroup_insubjredundantcsv_np == 'wangjinghong_clinicaldat')[0][0]
            else:
                for research_group_idx in range(len(researchgroup_insubjredundantcsv_np)):
                    research_group = researchgroup_insubjredundantcsv_np[research_group_idx]
                    if research_group in current_subj:
                        currentsubj_idx_inscanparam_csv = research_group_idx
                    break
        currentsubj_reptime = repetitiontimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_flipang = flipangelgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_echotime = echotimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentvoxelsize = voxelsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentslicenum = slicenumbergroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentmtxsize = matrixsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentfov = fovgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        if (currentsubj_reptime == 2530) \
            and (currentsubj_flipang == 7) \
            and (currentsubj_echotime == 3.65) \
            and (currentvoxelsize == '1*1*1') \
            and (currentslicenum == 224) \
            and (currentmtxsize == '176*256') \
            and (currentfov == '176*256'):
            if current_label == 0:
                pair3_hc_counter += 1
            elif current_label == 1:
                pair3_bd_counter += 1
            elif current_label == 2:
                pair3_mdd_counter += 1
            elif current_label == 3:
                pair3_sz_counter += 1
            elif current_label == 4:
                pair3_ocd_counter += 1
    print('HC num: %d' % pair3_hc_counter)
    print('BD num: %d' % pair3_bd_counter)
    print('MDD num: %d' % pair3_mdd_counter)
    print('SZ num: %d' % pair3_sz_counter)
    print('OCD num: %d' % pair3_ocd_counter)

    # calculate for pair 4:
    # repetition time: 2300
    # echo time: 3.5
    # flip angel: 9
    # FOV: 256*256
    # Matrix size: 256*256
    # number of slices: 192
    # slice thickness: 1
    # voxel size: 1*1*1
    pair4_hc_counter = 0
    pair4_bd_counter = 0
    pair4_mdd_counter = 0
    pair4_sz_counter = 0
    pair4_ocd_counter = 0
    print('##################### Pair4 is checking...')
    for idx_in_allinvolvedsubjcsv in range(len(subjgroup_inallinvolvedcsv_np)):
        current_subj = subjgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        current_label = labelgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        currensubj_researchgroupinallinvolvedcsv = researchgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        # if currentsubj in scanparam_csv, then record its parameters
        if current_subj in subjgroup_insubjredundantcsv_np:
            currentsubj_idx_inscanparam_csv = np.argwhere(subjgroup_insubjredundantcsv_np == current_subj)[0][0]
        # if currentsubj not in scanparam csv, then record the parameter of same research group
        else:
            if currensubj_researchgroupinallinvolvedcsv == 'WJH':
                currentsubj_idx_inscanparam_csv = np.argwhere(researchgroup_insubjredundantcsv_np == 'wangjinghong_clinicaldat')[0][0]
            else:
                for research_group_idx in range(len(researchgroup_insubjredundantcsv_np)):
                    research_group = researchgroup_insubjredundantcsv_np[research_group_idx]
                    if research_group in current_subj:
                        currentsubj_idx_inscanparam_csv = research_group_idx
                    break
        currentsubj_reptime = repetitiontimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_flipang = flipangelgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_echotime = echotimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentvoxelsize = voxelsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentslicenum = slicenumbergroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentmtxsize = matrixsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentfov = fovgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        if (currentsubj_reptime == 2300) \
            and (currentsubj_flipang == 9) \
            and (currentsubj_echotime == 3.5) \
            and (currentvoxelsize == '1*1*1') \
            and (currentslicenum == 192) \
            and (currentmtxsize == '256*256') \
            and (currentfov == '256*256'):
            if current_label == 0:
                pair4_hc_counter += 1
            elif current_label == 1:
                pair4_bd_counter += 1
            elif current_label == 2:
                pair4_mdd_counter += 1
            elif current_label == 3:
                pair4_sz_counter += 1
            elif current_label == 4:
                pair4_ocd_counter += 1
    print('HC num: %d' % pair4_hc_counter)
    print('BD num: %d' % pair4_bd_counter)
    print('MDD num: %d' % pair4_mdd_counter)
    print('SZ num: %d' % pair4_sz_counter)
    print('OCD num: %d' % pair4_ocd_counter)

    # calculate for pair 5:
    # repetition time: 2300
    # echo time: 2.96
    # flip angel: 9
    # FOV: 240*256
    # Matrix size: 240*256
    # number of slices: 192
    # slice thickness: 1
    # voxel size: 1*1*1
    pair5_hc_counter = 0
    pair5_bd_counter = 0
    pair5_mdd_counter = 0
    pair5_sz_counter = 0
    pair5_ocd_counter = 0
    print('##################### Pair5 is checking...')
    for idx_in_allinvolvedsubjcsv in range(len(subjgroup_inallinvolvedcsv_np)):
        current_subj = subjgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        current_label = labelgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        currensubj_researchgroupinallinvolvedcsv = researchgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        # if currentsubj in scanparam_csv, then record its parameters
        if current_subj in subjgroup_insubjredundantcsv_np:
            currentsubj_idx_inscanparam_csv = np.argwhere(subjgroup_insubjredundantcsv_np == current_subj)[0][0]
        # if currentsubj not in scanparam csv, then record the parameter of same research group
        else:
            if currensubj_researchgroupinallinvolvedcsv == 'WJH':
                currentsubj_idx_inscanparam_csv = np.argwhere(researchgroup_insubjredundantcsv_np == 'wangjinghong_clinicaldat')[0][0]
            else:
                for research_group_idx in range(len(researchgroup_insubjredundantcsv_np)):
                    research_group = researchgroup_insubjredundantcsv_np[research_group_idx]
                    if research_group in current_subj:
                        currentsubj_idx_inscanparam_csv = research_group_idx
                    break
        currentsubj_reptime = repetitiontimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_flipang = flipangelgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_echotime = echotimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentvoxelsize = voxelsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentslicenum = slicenumbergroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentmtxsize = matrixsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentfov = fovgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        if (currentsubj_reptime == 2300) \
            and (currentsubj_flipang == 9) \
            and (currentsubj_echotime == 2.96) \
            and (currentvoxelsize == '1*1*1') \
            and (currentslicenum == 192) \
            and (currentmtxsize == '240*256') \
            and (currentfov == '240*256'):
            if current_label == 0:
                pair5_hc_counter += 1
            elif current_label == 1:
                pair5_bd_counter += 1
            elif current_label == 2:
                pair5_mdd_counter += 1
            elif current_label == 3:
                pair5_sz_counter += 1
            elif current_label == 4:
                pair5_ocd_counter += 1
    print('HC num: %d' % pair5_hc_counter)
    print('BD num: %d' % pair5_bd_counter)
    print('MDD num: %d' % pair5_mdd_counter)
    print('SZ num: %d' % pair5_sz_counter)
    print('OCD num: %d' % pair5_ocd_counter)

    # calculate for pair 6:
    # repetition time: 2300
    # echo time: 2.96
    # flip angel: 9
    # FOV: 256*256
    # Matrix size: 256*256
    # number of slices: 192
    # slice thickness: 1
    # voxel size: 1*1*1
    pair6_hc_counter = 0
    pair6_bd_counter = 0
    pair6_mdd_counter = 0
    pair6_sz_counter = 0
    pair6_ocd_counter = 0
    print('##################### Pair6 is checking...')
    for idx_in_allinvolvedsubjcsv in range(len(subjgroup_inallinvolvedcsv_np)):
        current_subj = subjgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        current_label = labelgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        currensubj_researchgroupinallinvolvedcsv = researchgroup_inallinvolvedcsv_np[idx_in_allinvolvedsubjcsv]
        # if currentsubj in scanparam_csv, then record its parameters
        if current_subj in subjgroup_insubjredundantcsv_np:
            currentsubj_idx_inscanparam_csv = np.argwhere(subjgroup_insubjredundantcsv_np == current_subj)[0][0]
        # if currentsubj not in scanparam csv, then record the parameter of same research group
        else:
            if currensubj_researchgroupinallinvolvedcsv == 'WJH':
                currentsubj_idx_inscanparam_csv = np.argwhere(researchgroup_insubjredundantcsv_np == 'wangjinghong_clinicaldat')[0][0]
            else:
                for research_group_idx in range(len(researchgroup_insubjredundantcsv_np)):
                    research_group = researchgroup_insubjredundantcsv_np[research_group_idx]
                    if research_group in current_subj:
                        currentsubj_idx_inscanparam_csv = research_group_idx
                    break
        currentsubj_reptime = repetitiontimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_flipang = flipangelgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentsubj_echotime = echotimegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentvoxelsize = voxelsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentslicenum = slicenumbergroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentmtxsize = matrixsizegroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        currentfov = fovgroup_insubjredundantcsv_np[currentsubj_idx_inscanparam_csv]
        if (currentsubj_reptime == 2300) \
            and (currentsubj_flipang == 9) \
            and (currentsubj_echotime == 2.96) \
            and (currentvoxelsize == '1*1*1') \
            and (currentslicenum == 192) \
            and (currentmtxsize == '256*256') \
            and (currentfov == '256*256'):
            if current_label == 0:
                pair6_hc_counter += 1
            elif current_label == 1:
                pair6_bd_counter += 1
            elif current_label == 2:
                pair6_mdd_counter += 1
            elif current_label == 3:
                pair6_sz_counter += 1
            elif current_label == 4:
                pair6_ocd_counter += 1
    print('HC num: %d' % pair6_hc_counter)
    print('BD num: %d' % pair6_bd_counter)
    print('MDD num: %d' % pair6_mdd_counter)
    print('SZ num: %d' % pair6_sz_counter)
    print('OCD num: %d' % pair6_ocd_counter)


if __name__ == "__main__":
    allinvolvedsubj_csvpth = '/home/yxpt/Desktop/Projects/H_PDN_retrain_20211014/Files/AgeGender_dropWJHMDD_combine.csv'
    subjredundant_csvpth = '/home/yxpt/Desktop/Projects/H_PDN_retrain_20211014/Files/subjredundant_Scanparameters.csv'
    save_involved_scanparam_csvpth = '/home/yxpt/Desktop/Projects/H_PDN_retrain_20211014/Save/InvolvedSubjParam.csv'
    GetParam(allinvolvedsubj_csvpth,
             subjredundant_csvpth,
             save_involved_scanparam_csvpth)