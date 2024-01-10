class Config:
    EXP_NAME = "SWAG"
    OUTPUT_DIR = "output_swag"
    MIN_NUM_EPOCHS = 1
    MAX_NUM_EPOCHS = 35
    EARLY_STOPPING_EPOCHS = 3
    NUM_FOLDS = 2
    ONLY_EVAL_TEST_SET = False
    LOAD_OPTIMIZER = True
    DATA_DIR = "data/cinc2020_processed"
    TRAIN_DATA_DIR = "data/cinc2020_processed_tiny/train"
    TEST_DATA_DIR = "data/cinc2020_processed_tiny/test"
    LABEL_24 = "data/label_cinc2020_top24.csv"
    SWAG_EPOCHS = 30
    SWAG_UPDATE_FREQ = 1
    SWAG_LEARNING_RATE = 0.001
    DEVIATION_MATRIX_MAX_RANK = 15
    BMA_SAMPLES = 30
