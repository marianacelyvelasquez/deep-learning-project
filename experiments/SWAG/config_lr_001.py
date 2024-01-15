class Config:
    BMA_SAMPLES = 20
    DATA_DIR = "data/swag"
    DEVIATION_MATRIX_MAX_RANK = 15
    EARLY_STOPPING_EPOCHS = 3 # Doesnt matter
    EXP_NAME = "SWAG"
    LABEL_24 = "data/label_cinc2020_top24.csv"
    LOAD_OPTIMIZER = True
    MAX_NUM_EPOCHS = 5
    NUM_FOLDS = 2  # Training and validation set
    ONLY_EVAL_TEST_SET = False
    OUTPUT_DIR = "output_swag_lr_001"
    SWAG_LEARNING_RATE = 0.01
    SWAG_UPDATE_FREQ = 1
    TEST_DATA_DIR = "data/cinc2020_split_tiny/test"
    TRAIN_DATA_DIR = "data/cinc2020_split_tiny/train"
    DEVICE_NAME = "mps"
