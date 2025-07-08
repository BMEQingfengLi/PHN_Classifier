import pandas as pd
import os
import time
import csv
import numpy as np
from radiomics import featureextractor

def Generate_RadiomicsCSV(data_rootdir, subj_group_csvpth, DKAseg_txtpth, rdms_result_savepth):
    '''

    :param data_rootdir:
    :param subj_group_csvpth:
    :param DKAseg_txtpth:
    :param rdms_result_savepth:
    :return:
    '''
    # get dk cortex labels
    dk_table = pd.read_table(DKAseg_txtpth)
    dk_table_np = np.array(dk_table)
    ctx_label_list = []
    ctx_name_list = []
    for line in dk_table_np:
        current_row = line[0]
        current_row_element = current_row.split(' ')
        if current_row_element[1][:3] == 'ctx':
            ctx_label_list.append(int(current_row_element[0]))
            ctx_name_list.append(current_row_element[1][4:])

    # read available subj
    avail_subj_group_csv = pd.read_csv(subj_group_csvpth)
    avail_subj_group_np = np.array(avail_subj_group_csv['name'])

    # generate_radiomics feature
    rdms_csv = open(rdms_result_savepth, 'w', newline='')
    csv_writer = csv.writer(rdms_csv)
    first_row = ['name']
    firstrow_write_flag = False
    subj_counter = 1
    for subj in avail_subj_group_np:
        start_time = time.time()
        current_row = [subj]
        current_subjpth = os.path.join(data_rootdir, subj, 'rigid_Warped.nii.gz')
        current_dkatlaspth = os.path.join(data_rootdir, subj, 't1_DK_ApplyWarp.nii')
        print('%s  %d/%d' % (subj, subj_counter, len(avail_subj_group_np)))
        subj_counter += 1
        label_counter = 1
        for ctxlabel_idx in range(len(ctx_label_list)):
            ctxlabel = ctx_label_list[ctxlabel_idx]
            ctxname = ctx_name_list[ctxlabel_idx]
            extractor = featureextractor.RadiomicsFeatureExtractor()
            current_rdmsfeaturvector = extractor.execute(current_subjpth, current_dkatlaspth, label=ctxlabel)
            rdms_keys = current_rdmsfeaturvector.keys()
            if firstrow_write_flag == False:
                for rdms_key in rdms_keys:
                    if rdms_key[:8] == 'original':
                        first_row.append(ctxname + '_' + rdms_key)
            for rdms_key in rdms_keys:
                if rdms_key[:8] == 'original':
                    current_row.append(current_rdmsfeaturvector[rdms_key])
            print('Label: %s %d/%d time: %.2fs' % (ctxname, label_counter, len(ctx_label_list), time.time()-start_time))
            label_counter += 1
        if firstrow_write_flag == False:
            csv_writer.writerow(first_row)
            firstrow_write_flag = True
        csv_writer.writerow(current_row)
    rdms_csv.close()


if __name__ == '__main__':
    data_rootdir = "./Data_rootdir"
    subj_group_csvpth = './Files/availabledata.csv'
    DKAseg_txtpth = './Files/DKAseg_labels.txt'
    rdms_result_savepth = './Files/alldata_RDMS.csv'
    Generate_RadiomicsCSV(data_rootdir, subj_group_csvpth, DKAseg_txtpth, rdms_result_savepth)