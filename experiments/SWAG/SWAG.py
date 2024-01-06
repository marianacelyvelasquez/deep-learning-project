from experiments.SWAG.SWAGInference import SWAGInference

# 1. Load pre-pretrained model from checkpoint from dilatedCNN experiment

# 2. Collect k- weights from the model i.e. e.g. make 30 epochs
#    and collect 15 from them and build sgima thing

# 3. Calibrate

# 4. Evaluate


class SWAGExperiment:
    def __init__(self, swag_inference: SWAGInference) -> None:
        self.swag_inference = swag_inference

    def run(self) -> None:
        # Run SWAGInference::fit()
        self.swag_inference.fit_swag()

        # Run SWAGInference::calibrate()
        self.swag_inference.calibrate()

        # Run SWAGInference::evaluate()
        self.evaluate()

    def evaluate(self) -> None:
        # Implement your evaluation logic using the SWAGInference instance
        # You can access the necessary components through self.swag_inference
        pass
