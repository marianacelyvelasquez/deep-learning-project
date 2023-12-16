import enum
import torch

class InferenceMode(enum.Enum):
    """
    `MAP` predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0 #MARI I don't think we need this yet
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2

class SWAGInference(object):

    def __init__(
        self,
        train_xs: torch.Tensor,
        inference_mode: InferenceMode = InferenceMode.SWAG_DIAGONAL,

        #MARI some extra epochs to collect all the weights - 30 “different models“
        #MARI then you average over something something
        swag_epochs: int = 30,

        swag_learning_rate: float = 0.045,
        #MARI how often you update the swag statistics but we keep it as 1 (simple)
        #MARI build covariance stuff
        #MARI algo: bayesian model algorithm with SWAG
        swag_update_freq: int = 1, 

        #MARI how many columns your matrix D has at most (K in algo screenshot)
        deviation_matrix_max_rank: int = 15, 

        #MARI bma = Bayesian Model Averaging
        bma_samples: int = 30, 
    ):


        self.inference_mode = inference_mode
        self.swag_epochs = swag_epochs
        self.swag_learning_rate = swag_learning_rate
        self.swag_update_freq = swag_update_freq
        self.deviation_matrix_max_rank = deviation_matrix_max_rank
        self.bma_samples = bma_samples

        # Network used to perform SWAG. Currenlty not working
        self.network = dilated_CNN(in_channels=3, out_classes=6)

        # Store training dataset to recalculate batch normalization statistics during SWAG inference

        #MARI (allocate memory for) get trained dataset
        self.train_dataset = torch.utils.data.TensorDataset(train_xs) 

        # SWAG-diagonal
        # TODO(1): create attributes for SWAG-diagonal
        #  Hint: self._create_weight_copy() creates an all-zero copy of the weights
        #  as a dictionary that maps from weight name to values.
        #  Hint: you never need to consider the full vector of weights,
        #  but can always act on per-layer weights (in the format that _create_weight_copy() returns)

        # Full SWAG
        # TODO(2): create attributes for SWAG-diagonal
        #  Hint: check collections.deque

        # Calibration, prediction, and other attributes
        # TODO(2): create additional attributes, e.g., for calibration
        self._prediction_threshold = None  # this is an example, feel free to be creative
