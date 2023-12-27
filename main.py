import warnings

# We get a lot of warnings from sklearn.metrics._ranking (computing average precision)
# because we have lots of samples without any positive labels because the
# data has 111 classes but we only look at 24.
# We can ignore these warnings.
warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.metrics._ranking"
)

from experiments.dilated_CNN.dilatedCNN import dilatedCNNExperiment

if __name__ == "__main__":
    print("Running dialted_CNN experiment.")
    experiment = dilatedCNNExperiment()
    experiment.run()
