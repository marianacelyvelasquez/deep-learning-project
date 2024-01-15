class Config:
    DATA_DIR = "data/swag"
    EARLY_STOPPING_EPOCHS = 5
    EXP_NAME = "dilated_CNN"
    LABEL_24 = "data/label_cinc2020_top24.csv"
    LEARNING_RATE = 0.001
    LOAD_OPTIMIZER = False 
    MAX_NUM_EPOCHS = 35
    MIN_NUM_EPOCHS = 1
    NUM_FOLDS = 2
    ONLY_EVAL_TEST_SET = True
    OUTPUT_DIR = "dilated_nocv_epoch15_test"
    TEST_DATA_DIR = "data/swag/test"
    TRAIN_DATA_DIR = "data/swag/train"
    DEVICE = "cpu"
