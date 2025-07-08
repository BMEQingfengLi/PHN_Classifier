import csv
import pandas as pd
from scipy.stats import spearmanr
import numpy as np

def GetDemographicRow(idxlist, gender_group_np, age_group_np, eduage_group_np=None):
    '''

    :param idxlist:
    :param gender_group_np:
    :param age_group_np:
    :param eduage_group_np:
    :return:
    '''
    gender_age_eduage_list = []
    currentlabel_gendergroup_list = []
    currentlabel_agegroup_list = []
    currentlabel_eduage_group_list = []
    for idx in idxlist:
        currentlabel_gendergroup_list.append(gender_group_np[idx])
        currentlabel_agegroup_list.append(age_group_np[idx])
        if eduage_group_np != None:
            currentlabel_eduage_group_list.append(eduage_group_np[idx])

    # Male: 0, famale: 1
    currentlabel_gendergroup_np = np.array(currentlabel_gendergroup_list)
    male_amount = len(np.argwhere(currentlabel_gendergroup_np == 0))
    female_amount = len(np.argwhere(currentlabel_gendergroup_np == 1))
    gender_age_eduage_list.append('%d/%d' % (male_amount, female_amount))

    # Age: mean ± std (unbiased estimation: np.std(a, ddof = 1))
    currentlabel_agegroup_np = np.array(currentlabel_agegroup_list)
    age_mean = currentlabel_agegroup_np.mean()
    age_std = currentlabel_agegroup_np.std(ddof=1)
    gender_age_eduage_list.append('%.2f\u00B1%.2f' % (age_mean, age_std))

    if eduage_group_np != None:
        # Education age: mean ± std
        currentlabel_eduage_group_np = np.array(currentlabel_eduage_group_list)
        eduage_mean = currentlabel_eduage_group_np.mean()
        eduage_std = currentlabel_eduage_group_np.std(ddof=1)
        gender_age_eduage_list.append('%.2f\u00B1%.2f' % (eduage_mean, eduage_std))

    return gender_age_eduage_list


def GetDemographicCSV(csvfold_pth, demographic_csv_savepth):
    '''

    :param csvfold_pth:
    :param demographic_csv_savepth:
    :return:
    '''
    datacsv = pd.read_csv(csvfold_pth, sep=',')
    gender_group_np = np.array(datacsv['gender'])
    label_group_np = np.array(datacsv['label'])
    age_group_np = np.array(datacsv['age'])
    # eduage_group_np = np.array(datacsv['eduage'])

    hc_idxlist = np.argwhere(label_group_np == 0)[:, 0]
    bd_idxlist = np.argwhere(label_group_np == 1)[:, 0]
    mdd_idxlist = np.argwhere(label_group_np == 2)[:, 0]
    sz_idxlist = np.argwhere(label_group_np == 3)[:, 0]
    ocd_idxlist = np.argwhere(label_group_np == 4)[:, 0]

    new_csv = open(demographic_csv_savepth, 'w')
    csv_writer = csv.writer(new_csv)
    first_row = ['Characteristics',
                 'HC (n=%d)' % len(hc_idxlist),
                 'BD (n=%d)' % len(bd_idxlist),
                 'MDD (n=%d)' % len(mdd_idxlist),
                 'SZ (n=%d)' % len(sz_idxlist),
                 'OCD (n=%d)' % len(ocd_idxlist),
                 'R',
                 'p']
    csv_writer.writerow(first_row)

    hc_gender_age_eduage_list = GetDemographicRow(hc_idxlist, gender_group_np, age_group_np)
    bd_gender_age_eduage_list = GetDemographicRow(bd_idxlist, gender_group_np, age_group_np)
    mdd_gender_age_eduage_list = GetDemographicRow(mdd_idxlist, gender_group_np, age_group_np)
    sz_gender_age_eduage_list = GetDemographicRow(sz_idxlist, gender_group_np, age_group_np)
    ocd_gender_age_eduage_list = GetDemographicRow(ocd_idxlist, gender_group_np, age_group_np)

    gender_row = ['Gender (M/F)',
                  hc_gender_age_eduage_list[0],
                  bd_gender_age_eduage_list[0],
                  mdd_gender_age_eduage_list[0],
                  sz_gender_age_eduage_list[0],
                  ocd_gender_age_eduage_list[0],
                  spearmanr(gender_group_np, label_group_np)[0],
                  spearmanr(gender_group_np, label_group_np)[1]]
    csv_writer.writerow(gender_row)
    age_row = ['Age (Mean\u00B1Std)',
               hc_gender_age_eduage_list[1],
               bd_gender_age_eduage_list[1],
               mdd_gender_age_eduage_list[1],
               sz_gender_age_eduage_list[1],
               ocd_gender_age_eduage_list[1],
               spearmanr(age_group_np, label_group_np)[0],
               spearmanr(age_group_np, label_group_np)[1]]
    csv_writer.writerow(age_row)

    new_csv.close()


if __name__ == '__main__':
    csvfold_pth = './Files/T1available.csv'
    demographic_csv_savepth = './SaveFile/Demographic.csv'
    GetDemographicCSV(csvfold_pth, demographic_csv_savepth)