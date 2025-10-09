import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_end2end(nn.Module):
    def __init__(self, in_ch=1, out_channel=2):
        super(VGG_end2end, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm3d(64)
        self.relu6 = nn.ReLU(inplace=True)

        self.maxpool6 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv7= nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm3d(128)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm3d(128)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm3d(128)
        self.relu9 = nn.ReLU(inplace=True)

        self.maxpool9 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm3d(256)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm3d(256)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm3d(256)
        self.relu12 = nn.ReLU(inplace=True)

        self.maxpool12 = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.relubn1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(128, out_channel)

    def forward(self, x):
        #stage 1
        out1 = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))

        #stage 2
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.maxpool3(self.relu3(self.bn3(self.conv3(out2))))

        #stage 3
        out4 = self.relu4(self.bn4(self.conv4(out3)))
        out5 = self.relu5(self.bn5(self.conv5(out4)))
        out6 = self.maxpool6(self.relu6(self.bn6(self.conv6(out5))))

        #stage 4
        out7 = self.relu7(self.bn7(self.conv7(out6)))
        out8 = self.relu8(self.bn8(self.conv8(out7)))
        out9 = self.maxpool9(self.relu9(self.bn9(self.conv9(out8))))

        #stage 5
        out10 = self.relu10(self.bn10(self.conv10(out9)))
        out11 = self.relu11(self.bn11(self.conv11(out10)))
        out12 = self.maxpool12(self.relu12(self.bn12(self.conv12(out11))))

        out13 = out12.view(out12.size(0), -1)
        out14 = self.relubn1(self.fcbn1(self.fc1(out13)))
        out15 = self.fc2(out14)
        out_label = F.softmax(out15, dim=1)

        return out_label


class VGG_patch(nn.Module):
    def __init__(self, in_ch=1, out_channel=2):
        super(VGG_patch, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm3d(64)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm3d(64)
        self.relu6 = nn.ReLU(inplace=True)

        self.maxpool6 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv7= nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm3d(128)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm3d(128)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm3d(128)
        self.relu9 = nn.ReLU(inplace=True)

        self.maxpool9 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm3d(256)
        self.relu10 = nn.ReLU(inplace=True)

        self.conv11 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm3d(256)
        self.relu11 = nn.ReLU(inplace=True)

        self.conv12 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm3d(256)
        self.relu12 = nn.ReLU(inplace=True)

        self.maxpool12 = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.relubn1 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(128, out_channel)

    def forward(self, x):
        #stage 1
        out1 = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))

        #stage 2
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out3 = self.maxpool3(self.relu3(self.bn3(self.conv3(out2))))

        #stage 3
        out4 = self.relu4(self.bn4(self.conv4(out3)))
        out5 = self.relu5(self.bn5(self.conv5(out4)))
        out6 = self.maxpool6(self.relu6(self.bn6(self.conv6(out5))))

        #stage 4
        out7 = self.relu7(self.bn7(self.conv7(out6)))
        out8 = self.relu8(self.bn8(self.conv8(out7)))
        out9 = self.maxpool9(self.relu9(self.bn9(self.conv9(out8))))

        #stage 5
        out10 = self.relu10(self.bn10(self.conv10(out9)))
        out11 = self.relu11(self.bn11(self.conv11(out10)))
        out12 = self.maxpool12(self.relu12(self.bn12(self.conv12(out11))))

        out13 = out12.view(out12.size(0), -1)
        out14 = self.relubn1(self.fcbn1(self.fc1(out13)))
        out15 = self.fc2(out14)
        out_label = F.softmax(out15, dim=1)

        return out_label