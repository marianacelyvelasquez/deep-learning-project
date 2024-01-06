class Config:
    EXP_NAME = "dilated_CNN"
    MIN_NUM_EPOCHS = 1
    MAX_NUM_EPOCHS = 35
    EARLY_STOPPING_EPOCHS = 3
    NUM_FOLDS = 2
    ONLY_EVAL_TEST_SET = False
    LOAD_OPTIMIZER = True
    OUTPUT_DIR = "output"
    DATA_DIR = "data/cinc2020_processed_tiny"
    TRAIN_DATA_DIR = "data/cinc2020_processed_tiny/train"
    TEST_DATA_DIR = "data/cinc2020_processed_tiny/test"
    LABEL_24 = "data/label_cinc2020_top24.csv"
    LEARNING_RATE = 0.001