import pandas as pd
import numpy as np
import csv

def CopyRDMSData(origin_rdms_csvpth,
                 save_rdms_csvpth,
                 selected_subj_csvpth):
    '''

    :param origin_rdms_csvpth:
    :param save_rdms_csvpth:
    :param selected_subj_csvpth:
    :return:
    '''
    orig_rdms_csv = pd.read_csv(origin_rdms_csvpth)
    csv_keys = orig_rdms_csv.keys()
    subj_all_list_inorigcsv_np = np.array(orig_rdms_csv['name'])

    available_subj_csv = pd.read_csv(selected_subj_csvpth)
    available_subj_list_np = np.array(available_subj_csv['name'])

    new_csv = open(save_rdms_csvpth, 'w')
    csv_writer = csv.writer(new_csv)

    csv_writer.writerow(csv_keys.tolist())

    counter = 1
    for subj in available_subj_list_np:
        print('%s  %d/%d' % (subj, counter, len(available_subj_list_np)))
        subj_idx = np.argwhere(subj_all_list_inorigcsv_np == subj)[0][0]
        csv_writer.writerow(np.array(orig_rdms_csv)[subj_idx, :].tolist())
        counter += 1

    new_csv.close()


if __name__ == '__main__':
    origin_rdms_csvpth = "./Files/alldata_RDMS.csv"
    selected_subj_csvpth = './Files/availabledata.csv'
    save_rdms_csvpth = './Files/rdms.csv'
    CopyRDMSData(origin_rdms_csvpth,
                 save_rdms_csvpth,
                 selected_subj_csvpth)