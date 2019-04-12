import torch.nn as nn
import torch.nn.functional as F
import torch


class CorrespondenceModel(nn.Module):
    def __init__(self):
        super(CorrespondenceModel, self).__init__()
        self.conv_scan1 = nn.Conv3d(in_channels=1, out_channels=24, kernel_size=8, stride=2, padding = 4)
        self.conv_scan2 = nn.Conv3d(in_channels=24, out_channels=32, kernel_size=4, stride=2, padding = 2)
        self.conv_scan3 = nn.Conv3d(in_channels=32, out_channels=40, kernel_size=4, stride=2, padding = 2)
        self.conv_scan4 = nn.Conv3d(in_channels=40, out_channels=64, kernel_size=4, stride=2, padding = 1)

        self.conv_cad1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=2)
        self.conv_cad2 = nn.Conv3d(in_channels=4, out_channels=6, kernel_size=4, stride=2, padding=2)
        self.conv_cad3 = nn.Conv3d(in_channels=6, out_channels=8, kernel_size=4, stride=2, padding=1)


        self.bn_scan1 = nn.BatchNorm3d(24)
        self.bn_scan2 = nn.BatchNorm3d(32)
        self.bn_scan3 = nn.BatchNorm3d(40)
        self.bn_scan4 = nn.BatchNorm3d(64)

        self.bn_cad1 = nn.BatchNorm3d(4)
        self.bn_cad2 = nn.BatchNorm3d(6)
        self.bn_cad3 = nn.BatchNorm3d(8)

        self.feature_conv1 = nn.Conv3d(in_channels=72, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.feature_conv2 = nn.Conv3d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=1)

        self.feature_bn1 = nn.BatchNorm3d(40)
        self.feature_bn2 = nn.BatchNorm3d(40)

        self.scale_conv1 = nn.Conv3d(in_channels=40, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.scale_conv2 = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=4, stride=1)

        self.match_conv = nn.Conv3d(in_channels=40, out_channels=1, kernel_size=4, stride=1, padding = 0)

        self.heatmap_conv1 = nn.ConvTranspose3d(in_channels=40, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.heatmap_conv2 = nn.ConvTranspose3d(in_channels=32, out_channels=24, kernel_size=4, stride=2, padding=1)
        self.heatmap_conv3 = nn.ConvTranspose3d(in_channels=24, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.heatmap_conv4 = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)

        self.heatmap_bn1 = nn.BatchNorm3d(32)
        self.heatmap_bn2 = nn.BatchNorm3d(24)
        self.heatmap_bn3 = nn.BatchNorm3d(16)
        self.heatmap_bn4 = nn.BatchNorm3d(1)

    def forward(self, scan, cad):
        bs = scan.size()[0]
        s = F.relu(self.bn_scan1(self.conv_scan1(scan)))
        s = F.relu(self.bn_scan2(self.conv_scan2(s)))
        s = F.relu(self.bn_scan3(self.conv_scan3(s)))
        s = F.relu(self.bn_scan4(self.conv_scan4(s)))

        c = F.relu(self.bn_cad1(self.conv_cad1(cad)))
        c = F.relu(self.bn_cad2(self.conv_cad2(c)))
        c = F.relu(self.bn_cad3(self.conv_cad3(c)))

        feature = torch.cat([s, c], 1)
        feature = F.relu(self.feature_bn1(self.feature_conv1(feature)))
        feature = F.relu(self.feature_bn2(self.feature_conv2(feature)))

        scale_feat = F.relu(self.scale_conv1(feature))
        scale_pred = self.scale_conv2(scale_feat).reshape(bs, 3)

        match_pred = F.sigmoid(self.match_conv(feature)).reshape(bs)


        h = F.relu(self.heatmap_bn1(self.heatmap_conv1(feature)))
        h = F.relu(self.heatmap_bn2(self.heatmap_conv2(h)))
        h = F.relu(self.heatmap_bn3(self.heatmap_conv3(h)))
        h = self.heatmap_conv4(h)

        heatmap_pred = F.sigmoid(h)

        return match_pred, heatmap_pred, scale_pred


if __name__ == "__main__":

    scan = torch.rand(1,1,63,63,63).cuda()
    cad = torch.rand(1,1,32,32,32).cuda()

    model = CorrespondenceModel().cuda()
    print(model)
    match_pred, heatmap_pred, scale_pred = model(scan, cad)
    print(match_pred, scale_pred, heatmap_pred)

