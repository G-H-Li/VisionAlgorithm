import numpy as np
import torch
from PIL import Image
from imageio import imsave
from algorithm import cutout, ShearX

data_path = "./data/"
output_path = "./output/"


def readImg(filename, is_array=True):
    """
    加载图片
    """
    file_path = data_path + filename

    img = Image.open(file_path)
    # img.show()
    if is_array:
        img_array = np.array(img)
        return img_array
    else:
        return img


def outputImg(filename, img_array):
    """
    保存并展示图片
    """
    file_path = output_path + filename
    imsave(file_path, img_array)
    # plt.imshow(img_array)


def cutout_test():
    inArray = readImg("test1.jpg")
    tensor = torch.from_numpy(inArray)
    for i in range(0, 16, 2):
        out = cutout(tensor, i)
        out = out.numpy().astype(np.uint8)
        fileName = str(i) + '.jpg'
        outputImg(fileName, out)


def common_test(func=ShearX):
    out = func
    Image._show(out)


if __name__ == "__main__":
    print("test start")
    img = readImg("test1.jpg", False)
    common_test(ShearX(img, -0.2))
    print("test end")
