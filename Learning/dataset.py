import sys
assert sys.version_info >= (3,5)

import sys
sys.path.append("../Routines/LoadVox")
sys.path.append("../Routines/Script")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import LoadVox
import JSONHelper
import torch.utils.data as data

import os

class Scan2CADdataset(data.Dataset):
    def __init__(self,root):
        self.dataset = JSONHelper.read(os.path.join(root, 'trainset.json'))

        self.dataset_filtered = []
        for f in self.dataset:
            center_path = f['center'][3:]
            heatmap_path = f['heatmap'][3:]
            if os.path.exists(center_path) and os.path.exists(heatmap_path):
                self.dataset_filtered.append(f)

        self.dataset = self.dataset_filtered
        self.positive_matches = len(self.dataset)
        self.root = root

    def __len__(self):
        return len(self.dataset) * 11

    def get_positive_data(self, idx):
        path = self.dataset[idx]
        center_path = path['center'][3:]
        heatmap_path = path['heatmap'][3:]
        scale = path['scale']
        match = bool(path['match'])

        sdfpdf = LoadVox.load_vox_with_pdf_np(os.path.join(heatmap_path))
        CAD_sdf = sdfpdf[:32 ** 3].reshape((32, 32, 32))
        CAD_pdf = sdfpdf[32 ** 3:].reshape((32, 32, 32))

        sdf = LoadVox.load_vox_np(os.path.join(center_path))
        scan_sdf = sdf.reshape((63, 63, 63))

        return scan_sdf, CAD_sdf, CAD_pdf, scale, match

    def __getitem__(self, idx):
        if idx < self.positive_matches:
            return self.get_positive_data(idx)
        else:
            pos_idx = np.random.randint(0, self.positive_matches)
            neg_idx = pos_idx
            while neg_idx == pos_idx:
                neg_idx = np.random.randint(0, self.positive_matches)

            center_path = self.dataset[pos_idx]['center'][3:]
            heatmap_path = self.dataset[neg_idx]['heatmap'][3:]

            sdfpdf = LoadVox.load_vox_with_pdf_np(os.path.join(heatmap_path))
            CAD_sdf = sdfpdf[:32 ** 3].reshape((32, 32, 32))

            sdf = LoadVox.load_vox_np(os.path.join(center_path))
            scan_sdf = sdf.reshape((63, 63, 63))
            return scan_sdf, CAD_sdf, None, None, False

if __name__ == '__main__':
    dataset = Scan2CADdataset(sys.argv[1])

    for i in range(100):
        scan_sdf, CAD_sdf, CAD_pdf, scale, match = dataset[np.random.randint(len(dataset))]
        print(np.sum(np.abs(scan_sdf) < 0.05), np.sum(np.abs(CAD_sdf) < 0.5), np.sum(CAD_pdf))


    exit()
    sdfpdf = LoadVox.load_vox_with_pdf_np('/home/fei/Development/Scan2CAD/Assets/training-data/CAD-heatmaps-sample/scene0291_01_03001627_8ab6783b1dfbf3a8a5d9ad16964840ab_13_2.vox2')
    sdf = sdfpdf[:32**3].reshape((32,32,32))
    pdf = sdfpdf[32**3:].reshape((32,32,32))

    voxel = (np.abs(sdf) < 0.5).astype(np.float32)
    colors = np.empty(voxel.shape, dtype=object)
    colors[voxel > 0] = 'blue'
    colors[pdf > 0] = 'red'

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, facecolors=colors, alpha=0.7)
    plt.show()

    sdf = LoadVox.load_vox_np('/home/fei/Development/Scan2CAD/Assets/training-data/scan-centers-sample/scene0291_01_03001627_8ab6783b1dfbf3a8a5d9ad16964840ab_13_2.vox')
    sdf = sdf.reshape((63,63,63))

    voxel = (np.abs(sdf) < 0.05).astype(np.float32)

    colors = np.empty(voxel.shape, dtype=object)
    colors[voxel > 0] = 'blue'
    colors[30:33, 30:33, 30:33] = 'red'
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel, facecolors=colors, alpha=0.7, edgecolor='k')

    plt.show()
