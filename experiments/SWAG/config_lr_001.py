class Config:
    BMA_SAMPLES = 30
    DATA_DIR = "data/swag"
    DEVIATION_MATRIX_MAX_RANK = 15
    EARLY_STOPPING_EPOCHS = 3
    EXP_NAME = "SWAG"
    LABEL_24 = "data/label_cinc2020_top24.csv"
    LOAD_OPTIMIZER = True
    MAX_NUM_EPOCHS = 15
    NUM_FOLDS = 2 #Training and validation set
    ONLY_EVAL_TEST_SET = False
    OUTPUT_DIR = "output_swag_lr_001"
    SWAG_LEARNING_RATE = 0.001
    SWAG_UPDATE_FREQ = 1
    TEST_DATA_DIR = "data/swag/test"
    TRAIN_DATA_DIR = "data/swag/train"
    DEVICE_NAME = "cuda:1"
