import numpy as np
import torch
from PIL import Image
from imageio import imsave
from algorithm import cutout

data_path = "./data/"
output_path = "./output/"


def readImg(filename):
    """
    加载图片
    """
    file_path = data_path + filename

    img = Image.open(file_path)
    # img.show()
    img_array = np.array(img)
    return img_array


def outputImg(filename, img_array):
    """
    保存并展示图片
    """
    file_path = output_path + filename
    imsave(file_path, img_array)
    # plt.imshow(img_array)


if __name__ == "__main__":
    print("test start")
    inArray = readImg("test1.jpg")
    tensor = torch.from_numpy(inArray)
    for i in range(0, 16, 2):
        out = cutout(tensor, i)
        out = out.numpy().astype(np.uint8)
        fileName = str(i) + '.jpg'
        outputImg(fileName, out)
    print("test end")
