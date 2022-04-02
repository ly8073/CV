import os.path

import matplotlib.pyplot as plt
import torch
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


def identify_number():
    test_set = DataProcess(cfg.FOLDER, cfg.TEST_IMAGES, cfg.TEST_LABELS, onehot=False,
                           transform=transforms.ToTensor())
    check_points = os.path.join(cfg.FOLDER, f"checkpoints_{cfg.EPOCH - 1}.pkl")
    nets = torch.load(check_points).eval().cpu()
    total, correct = 0, 0
    while True:
        try:
            test_img_number = int(input("input a number(0~999):"))
            image, label = test_set[test_img_number]
            y = nets(image)
            prop, target = torch.max(y, dim=0)
            print(f"label={label}\n"
                  f"judge={target.item()}, props={prop.item()}")
            plt.imshow(image.squeeze())
            plt.show()
            total += 1
            if target.item() == label:
                correct += 1
        except Exception:
            break
    print(f"tested {total} pictures, {correct} correct, rate={100 * correct / total}%")


if __name__ == "__main__":
    main()
    identify_number()