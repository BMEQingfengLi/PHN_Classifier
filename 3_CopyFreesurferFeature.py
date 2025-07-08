import pandas as pd
import numpy as np
import csv

def CpFSfeature(origin_fsfeature_csvpth,
                selected_subj_csvpth,
                save_fs_csvpth):
    '''

    :param origin_fsfeature_csvpth:
    :param selected_subj_csvpth:
    :param save_fs_csvpth:
    :return:
    '''
    selected_subj_csv = pd.read_csv(selected_subj_csvpth)
    available_subj_np = np.array(selected_subj_csv['name'])

    origin_fsfeature_csv = pd.read_csv(origin_fsfeature_csvpth)
    csv_keys = origin_fsfeature_csv.keys()
    subj_list_inorigcsv_np = np.array(origin_fsfeature_csv['name'])

    new_csv = open(save_fs_csvpth, 'w')
    csv_writer = csv.writer(new_csv)
    csv_writer.writerow(csv_keys.tolist())

    counter = 1
    for subj in available_subj_np:
        print('%s  %d/%d' % (subj, counter, len(available_subj_np)))
        subj_idx = np.argwhere(subj_list_inorigcsv_np == subj)[0][0]
        csv_writer.writerow(np.array(origin_fsfeature_csv)[subj_idx])
        counter += 1

    new_csv.close()


if __name__ == '__main__':
    origin_fsfeature_csvpth = "./Files/DatasetFeatureTable_T1.csv"
    selected_subj_csvpth = './Files/availabledata.csv'
    save_fs_csvpth = './Files/fsfeature.csv'
    CpFSfeature(origin_fsfeature_csvpth,
                selected_subj_csvpth,
                save_fs_csvpth)