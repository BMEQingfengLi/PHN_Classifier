import csv
import os
import copy
import pandas as pd
import numpy as np
from random import shuffle

def CSVCroValSplit(origin_csv_pth, croval_fold, csv_savedir):
    '''

    :param origin_csv_pth:
    :param croval_fold:
    :param csv_savedir:
    :return:
    '''
    random.seed(42)

    origin_csv = pd.read_csv(origin_csv_pth)
    origin_name_list_np = np.array(origin_csv['name'])
    origin_gender_list_np = np.array(origin_csv['gender'])
    origin_label_list_np = np.array(origin_csv['label'])
    origin_age_list_np = np.array(origin_csv['age'])

    # randomly shuffle data
    idx_list = list(range(len(origin_name_list_np)))
    shuffle(idx_list)

    # split cross-validation bags
    idx_range_of_idx_list = []
    bag_length = int(len(idx_list) / croval_fold)
    for idx in range(1, croval_fold+1):
        if not 0 < abs(idx * bag_length - len(idx_list)) < bag_length:
            idx_range_of_idx_list.append([(idx-1) * bag_length, idx * bag_length])
        else:
            idx_range_of_idx_list.append([(idx - 1) * bag_length, len(idx_list)])

    # save subject line as 3:1
    for cv_idx in range(1, croval_fold+1):
        save_cv_traincsv_pth = os.path.join(csv_savedir, '%dfoldcv_fold%d_train.csv' % (croval_fold, cv_idx))
        save_cv_valcsv_pth = os.path.join(csv_savedir, '%dfoldcv_fold%d_val.csv' % (croval_fold, cv_idx))
        save_cv_testcsv_pth = os.path.join(csv_savedir, '%dfoldcv_fold%d_test.csv' % (croval_fold, cv_idx))
        traincsv = open(save_cv_traincsv_pth, 'w')
        valcsv = open(save_cv_valcsv_pth, 'w')
        testcsv = open(save_cv_testcsv_pth, 'w')
        train_csv_writer = csv.writer(traincsv)
        val_csv_writer = csv.writer(valcsv)
        test_csv_writer = csv.writer(testcsv)
        first_row = ['name', 'gender', 'label', 'age']
        train_csv_writer.writerow(first_row)
        val_csv_writer.writerow(first_row)
        test_csv_writer.writerow(first_row)

        test_idx_range = idx_range_of_idx_list[cv_idx-1]
        trainandval_rangelist = copy.deepcopy(idx_range_of_idx_list)
        trainandval_rangelist.remove(test_idx_range)
        trainandval_range = []
        for tvrange in trainandval_rangelist:
            for idx in range(tvrange[0], tvrange[1]):
                trainandval_range.append(idx)
        test_idx_range = list(range(test_idx_range[0], test_idx_range[1]))

        for current_idx in range(len(trainandval_range)):
            if current_idx < (len(trainandval_range) * 3 / 4):
                idx = idx_list[trainandval_range[current_idx]]
                current_row = [origin_name_list_np[idx],
                               origin_gender_list_np[idx],
                               origin_label_list_np[idx],
                               origin_age_list_np[idx]]
                train_csv_writer.writerow(current_row)
            else:
                idx = idx_list[trainandval_range[current_idx]]
                current_row = [origin_name_list_np[idx],
                               origin_gender_list_np[idx],
                               origin_label_list_np[idx],
                               origin_age_list_np[idx]]
                val_csv_writer.writerow(current_row)
        for current_idx in range(len(test_idx_range)):
            idx = idx_list[test_idx_range[current_idx]]
            current_row = [origin_name_list_np[idx],
                           origin_gender_list_np[idx],
                           origin_label_list_np[idx],
                           origin_age_list_np[idx]]
            test_csv_writer.writerow(current_row)
        traincsv.close()
        valcsv.close()
        testcsv.close()


if __name__ == '__main__':
    origin_csv_pth = './Files/availabledata.csv'
    croval_fold = 5
    csv_savedir = './CrossValData'
    CSVCroValSplit(origin_csv_pth, croval_fold, csv_savedir)