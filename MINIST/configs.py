import torch.cuda

BATCH_SIZE = 512
EPOCH = 20
LEARNING_RATE = 1e-3

# DURING TRAIN
LOG_PERIOD = 100
CHECKPOINT = 5

# DATAS
FOLDER = r"D:\02 Coding\05 DataBase\01 MINIST"
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
CHECKPOINT_FOLDER = r"./checkpoints"
