import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import std, mean, sqrt
import mpl_toolkits.axisartist as axisart


def ImagePlot(compareresult_rootdir,
              resultimg_savepth):
    '''

    :param compareresult_rootdir:
    :param resultimg_savepth:
    :return:
    '''
    figure, ax = plt.subplots(8, 2, figsize=(80, 80))

    # load results for different pairs
    fig_coordinate_idx = 0
    fig_coordinate_list = [[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1],
                           [2, 0],
                           [2, 1],
                           [3, 0],
                           [3, 1],
                           [4, 0],
                           [4, 1],
                           [5, 0],
                           [5, 1],
                           [6, 0],
                           [6, 1],
                           [7, 0],
                           [7, 1]]
    # metric_list = ['ACC', 'SEN', 'SPE', 'AUC', 'F1', 'MCC']
    metric_list = ['ACC', 'SEN', 'SPE', 'AUC']
    method_list = ['Logistic Regression', 'SVM', 'XGBoost', 'GBDT', 'Random Forest', 'VGG16', 'ResNet18', 'PHNLSTMagegen']
    method_list_forfigplot = ['Logistic Regression', 'SVM', 'XGBoost', 'GBDT', 'Random Forest', 'VGG', 'ResNet', 'PHN']
    # plt.style.use('fast')
    for pairs in ['BDHC',
                  'BDMDD',
                  'BDSZ',
                  'BDOCD',
                  'MDDHC',
                  'MDDSZ',
                  'MDDOCD',
                  'SZHC',
                  'SZOCD',
                  'OCDHC',
                  'HCOthers',
                  'BDOthers',
                  'MDDOthers',
                  'SZOthers',
                  'OCDOthers']:
        result_csvpth = os.path.join(compareresult_rootdir, pairs, 'SMHC5foldresult.csv')
        result_csv_np = np.array(pd.read_csv(result_csvpth))

        # read values according to metrics
        metric_start_xlocation = np.array(list(range(len(metric_list))))
        metric0_start_xlocation = np.array(list(range(len(metric_list))))

        fig_coordinate = fig_coordinate_list[fig_coordinate_idx]
        fig_rowcounter = fig_coordinate[0]
        fig_columncounter = fig_coordinate[1]

        for method_idxinlist in range(len(method_list)):
            method = method_list[method_idxinlist]
            method_forplot = method_list_forfigplot[method_idxinlist]
            current_metric_meanlist_foreachmethod = []
            current_metric_stdlist_foreachmethod = []
            method_row_idx = np.argwhere(result_csv_np[:, 0] == method)[0][0]
            for metric_idx in range(len(metric_list)):
                current_metric_meanandstd_str = result_csv_np[method_row_idx][metric_idx + 1]
                surrent_metric_mean_float = float(current_metric_meanandstd_str.split('\u00B1')[0])
                surrent_metric_std_float = float(current_metric_meanandstd_str.split('\u00B1')[1])
                current_metric_meanlist_foreachmethod.append(surrent_metric_mean_float)
                current_metric_stdlist_foreachmethod.append(surrent_metric_std_float)
            ax[fig_rowcounter][fig_columncounter].bar(metric_start_xlocation,
                                   current_metric_meanlist_foreachmethod,
                                   yerr=current_metric_stdlist_foreachmethod,
                                   error_kw={'ecolor' : '0.2', 'capsize' :6},
                                   alpha=0.9,
                                   width=float(1)/(len(method_list)+1),
                                   label=method_forplot,
                                   edgecolor='black')
            metric_start_xlocation = metric_start_xlocation + float(1)/(len(method_list) + 1)

        # custom xaxis
        ax[fig_rowcounter][fig_columncounter].set_xticks(metric0_start_xlocation + (float(1)/(len(method_list)+1)) * float(len(method_list))/2)
        ax[fig_rowcounter][fig_columncounter].set_xticklabels(metric_list)

        # plot legends
        ax[fig_rowcounter][fig_columncounter].tick_params(labelsize=20)
        ax[fig_rowcounter][fig_columncounter].set_ylim(0, 1)
        ax[fig_rowcounter][fig_columncounter].set_xlabel('Metric', fontdict={'weight': 'normal', 'size': 30})
        ax[fig_rowcounter][fig_columncounter].set_ylabel('Value', fontdict={'weight': 'normal', 'size': 30})
        ax[fig_rowcounter][fig_columncounter].plot([-0.1, metric_start_xlocation[-1]], [0, 0], color='black')
        ax[fig_rowcounter][fig_columncounter].grid(axis='y', ls='--')
        ax[fig_rowcounter][fig_columncounter].set_axisbelow('True')
        ax[fig_rowcounter][fig_columncounter].set_title(pairs, fontsize=50)
        fig_coordinate_idx += 1

    ax[7][0].legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0, fontsize=40)
    plt.axis('off')  # hide not necessary subfigure
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['savefig.dpi'] = 500
    plt.tight_layout()
    plt.savefig(resultimg_savepth, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    compareresult_rootdir = './CompareResult_withDL/'
    resultimg_savepth = './Save/BinaryCLF_MethodComparison.svg'
    ImagePlot(compareresult_rootdir,
              resultimg_savepth)