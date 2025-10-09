import pandas as pd
import numpy as np
import csv

def GenerateComparison(clinicdiag_csvpth,
                       expertdiag_csvpth,
                       save_comparison_csvpth):
    '''

    :param clinicdiag_csvpth:
    :param expertdiag_csvpth:
    :param save_comparison_csvpth:
    :return:
    '''
    # load clinic diagnosis
    clinicdiag_csv = pd.read_csv(clinicdiag_csvpth, encoding='gbk')
    idgroup_inclinicsv_np = np.array(clinicdiag_csv['ApplyID'])
    subjgroup_inclinicsv_np = np.array(clinicdiag_csv['subj'])
    diaggroup_inclinicsv_np = np.array(clinicdiag_csv['diagnosis'])

    # load ezpert diagnosis
    expertdiag_csv = pd.read_csv(expertdiag_csvpth, encoding='gbk')
    idgroup_inexpertdiagcsv_np = np.array(expertdiag_csv['applyID'])
    diaggroup_inexpertdiagcsv_np = np.array(expertdiag_csv['ExpertDiag'])

    # write comparison csv
    comparison_csv = open(save_comparison_csvpth, 'w', encoding='gbk')
    csv_writer = csv.writer(comparison_csv)
    first_row = ['applyID', 'subj', 'clinicdiag', 'expertdiag']
    csv_writer.writerow(first_row)
    for idx_inexpertdiagcsv in range(len(idgroup_inexpertdiagcsv_np)):
        current_id = idgroup_inexpertdiagcsv_np[idx_inexpertdiagcsv]
        current_expertdiag = diaggroup_inexpertdiagcsv_np[idx_inexpertdiagcsv]
        if current_id in idgroup_inclinicsv_np:
            idx_inclinicdiagcsv = np.argwhere(idgroup_inclinicsv_np == current_id)[0][0]
            subj = subjgroup_inclinicsv_np[idx_inclinicdiagcsv]
            current_clinicdiag = diaggroup_inclinicsv_np[idx_inclinicdiagcsv]
            current_row = [current_id, subj, current_clinicdiag, current_expertdiag]
            csv_writer.writerow(current_row)
    comparison_csv.close()


if __name__ == '__main__':
    clinicdiag_csvpth = './ClinicalDiagnosis.csv'
    expertdiag_csvpth = './ExpertDiag.csv'
    save_comparison_csvpth = './Save/CompareClinicAndExpert.csv'
    GenerateComparison(clinicdiag_csvpth,
                       expertdiag_csvpth,
                       save_comparison_csvpth)