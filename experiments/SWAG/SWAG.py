import enum
import torch
import typing
import collections


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
        # then you average over something something
        swag_epochs: int = 30,

        swag_learning_rate: float = 0.045,
        #MARI how often (in #epochs) you update the swag statistics - we keep it as 1 (simple)
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
        self.weight_copy = self._create_weight_copy()
        # MARI : did Hint1, must understand if Hint2 is done as well or another tricky thing
        #  Hint 2: you never need to consider the full vector of weights,
        #  but can always act on per-layer weights (in the format that _create_weight_copy() returns)

        # Full SWAG
        self.weight_copies_full = collections.deque(maxlen=swag_epochs * swag_update_freq)
        # MARI: This deque will store multiple weight copies over the course of training

        # Calibration, prediction, and other attributes
        # TODO(2): create additional attributes, e.g., for calibration
        self._prediction_threshold = None  # this is an example, feel free to be creative


    def update_swag(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network. (here the CNN)
        """

        # Create a copy of the current network weights
        # MARI PASCAL TODO: check that this is compatible with the CNN implementation (named_parameters field)
        current_params = {name: param.detach() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        for name, param in current_params.items():
            # TODO(1): update SWAG-diagonal attributes for weight `name` using `current_params` and `param`
            raise NotImplementedError("Update SWAG-diagonal statistics")

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            # TODO(2): update full SWAG attributes for weight `name` using `current_params` and `param`
            raise NotImplementedError("Update full SWAG statistics")


    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False) # dictionary {name: zeroes-matrix}
            for name, param in self.network.named_parameters()
            #MARI get weights; tensors usually are 4-dimensional [dim 1: #samples, dim 2: channels, dims 3-4: kernel size]
        }
    
    def update_swag(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        current_params = {name: param.detach() for name, param in self.network.named_parameters()}

        # SWAG-diagonal
        for name, param in current_params.items():
            #MARI TODO: update SWAG-diagonal attributes for weight `name` using `current_params` and `param`
            raise NotImplementedError("Update SWAG-diagonal statistics")

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            self._update_full_swag(current_params)

    def _update_full_swag(self, current_params) -> None:
        """Update the Full SWAG weight copies during training."""
        # Append a copy of the current weights to the deque
        self.weight_copies_full.append(current_params)

