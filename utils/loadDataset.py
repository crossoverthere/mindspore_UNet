import os
import cv2
import numpy as np


class Dataset:
    def __init__(self, dataDir="../data/train/"):
        self.Dir = dataDir
        self.data = os.listdir(os.path.join(dataDir, "image"))
        self.mask = os.listdir(os.path.join(dataDir, "mask"))

    def __getitem__(self, item):
        image = cv2.imread(os.path.join(self.Dir, "image/", self.data[item]))
        mask = cv2.imread(os.path.join(self.Dir, "mask/", self.mask[item]))
        # 保留一个通道，后续并入图像预处理中
        image = image[:, :, 0].astype(np.float32)
        mask = mask[:, :, 0].astype(np.float32)
        # 做一个简单的归一化，后续并入图像预处理
        image = image / 255
        mask = mask / 255
        # 调整图像通道数, 我为什么要在这里调整？
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        image = np.reshape(image, [1, 512, 512])
        mask = np.reshape(mask, [1, 512, 512])
        return image, mask

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data = Dataset()