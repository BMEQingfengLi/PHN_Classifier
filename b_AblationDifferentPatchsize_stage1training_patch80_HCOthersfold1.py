from Utils.Stage1_PFENTrainingProcess import Stage1Training
import argparse

# training for VGG / ResNet (as stage 1 of 2-stage)

def main():
    '''

    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgrootpth',
                        default="/HuaweiData/sharehome/yxpt/homeDocument/Documents/DL_trainingdata/20220809_PHN_Data/SMHC",
                        type=str,
                        help='input image rootpath (including training and validation data)')
    parser.add_argument('--gpuidx',
                        default=0,
                        type=int,
                        help='Index of GPU. The training process uses a single GPU.')
    parser.add_argument('--savepth',
                        default='./AblationHCOthers_DifferentPatchSize/Patch80_HCOthers_stage1_fold1',
                        type=str,
                        help='model and results save path')
    parser.add_argument('--epoch',
                        default=500,
                        type=int,
                        help='training epoch')
    parser.add_argument('--batchsize',
                        default=8,
                        type=int,
                        help='training batchsize')
    parser.add_argument('--threadnum',
                        default=5,
                        type=int,
                        help='number of thread')
    parser.add_argument('--patchsize',
                        default=80,
                        type=int,
                        help='stage1 image patch size')
    parser.add_argument('--imgname',
                        default='t1_brain.nii.gz',
                        type=str,
                        help='input image name')
    parser.add_argument('--traincsvpth',
                        default='./CrossValData/5foldcv_fold1_train.csv',
                        type=str,
                        help='training csv path')
    parser.add_argument('--valcsvpth',
                        default='./CrossValData/5foldcv_fold1_val.csv',
                        type=str,
                        help='validation csv path')
    parser.add_argument('--pretrainmodelpth',
                        default='',
                        type=str,
                        help='pretrain model path')
    parser.add_argument('--label0',
                        default=[1,2,3,4],
                        type=list,
                        help='focused label 0 list')
    parser.add_argument('--label1',
                        default=[0],
                        type=list,
                        help='focused label 1 list')

    args = parser.parse_args()

    # basic setting
    imgrootpth = args.imgrootpth
    gpuidx = args.gpuidx
    savepth = args.savepth
    epoch = args.epoch
    batchsize = args.batchsize
    threadnum = args.threadnum
    patchsize = args.patchsize
    imgname = args.imgname
    traincsvpth = args.traincsvpth
    valcsvpth = args.valcsvpth
    pretrainmodelpth = args.pretrainmodelpth
    label0_list = args.label0
    label1_list = args.label1

    # Start training process
    Stage1Training(imgrootpth,
                gpuidx,
                savepth,
                epoch,
                batchsize,
                threadnum,
                patchsize,
                imgname,
                traincsvpth,
                valcsvpth,
                pretrainmodelpth,
                label0_list,
                label1_list)


if __name__ == '__main__':
    main()
