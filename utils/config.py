class Config:
    DATA_DIR = "data/swag"
    EARLY_STOPPING_EPOCHS = 3
    EXP_NAME = "dilated_CNN"
    LABEL_24 = "data/label_cinc2020_top24.csv"
    LEARNING_RATE = 0.001
    LOAD_OPTIMIZER = True
    MAX_NUM_EPOCHS = 35
    MIN_NUM_EPOCHS = 1
    NUM_FOLDS = 2
    ONLY_EVAL_TEST_SET = False
    OUTPUT_DIR = "outputJan10"
    TEST_DATA_DIR = "data/swag/test"
    TRAIN_DATA_DIR = "data/swag/train"