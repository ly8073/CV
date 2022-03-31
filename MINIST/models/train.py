from torchvision.transforms import transforms

from MINIST.models.model_zoo import NetWork
from MINIST.DataPreprocess.DealData import DataProcess


def train():
    nets = NetWork(1, 16)
    folder = r"D:\02 Coding\05 DataBase\01 MINIST"
    train_set = DataProcess(folder, 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                            transform=transforms.ToTensor())
    test_set = DataProcess(folder, 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
                           transform=transforms.ToTensor())
    