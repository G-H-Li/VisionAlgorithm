import numpy as np
import torch


def cutout(img_array, length):
    """
    此方法主要是在图像上增加一条纵向分割线，线的宽度与length有关，如output/cutout.jpg
    """
    # 注：此处采用张量计算，张量与矩阵存在差异
    h, w = img_array.size(1), img_array.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)

    mask[y1:y2, x1:x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img_array)
    img_array = img_array * mask  # 张量乘法，img_array *= mask这种写法存在问题
    return img_array
