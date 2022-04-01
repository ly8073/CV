from torchvision.transforms import transforms

from MINIST.models.train import train
from MINIST.DataPreprocess.DealData import DataProcess
import MINIST.configs as cfg


def main():
    train_set = DataProcess(cfg.FOLDER, cfg.TRAIN_IMAGES, cfg.TRAIN_LABELS,
                            transform=transforms.ToTensor())
    test_set = DataProcess(cfg.FOLDER, cfg.TEST_IMAGES, cfg.TEST_LABELS,
                           transform=transforms.ToTensor())
    train(train_set, test_set)


if __name__ == "__main__":
    main()