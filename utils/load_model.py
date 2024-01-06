import torch
from experiments.dilated_CNN.dilatedCNN import dilatedCNNExperiment  # Adjust the import path based on your project structure
from utils.ExperimentConfig import ExperimentConfig

def load_model(
        checkpoint_path,
        X_train, #beginning: for dilatedCNN
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        CV_k,
        ):
    experiment_config = ExperimentConfig(X_train, y_train, X_val, y_val, X_test, y_test, CV_k, checkpoint_path)

    model = dilatedCNNExperiment(experiment_config)  # Initialize your model here
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Add more loading logic as needed
    return model