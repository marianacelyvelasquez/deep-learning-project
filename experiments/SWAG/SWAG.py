import enum
import torch
import typing
import collections
import tqdm

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
        self.theta = self._create_weight_copy()
        self.theta_squared = self._create_weight_copy()


        # Full SWAG
        #MARI-PASCAL: weight_copies_full is the D-matrix
        self.weight_copies_full = collections.deque(maxlen=swag_epochs * swag_update_freq) #Pascal's team did different
        # MARI: This deque will store multiple weight copies over the course of training
        # MARI-PASCAL: possibly use something different instead of deque

        #MARI TODO:
        # Calibration, prediction, and other attributes
        # TODO(2): create additional attributes, e.g., for calibration
        
        self._prediction_threshold = None  # this is an example, feel free to be creative
        self.n = 0  # number of samples seen so far


    #Initialization
    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False) # dictionary {name: zeroes-matrix}
            for name, param in self.network.named_parameters()
            #MARI get weights; tensors usually are 4-dimensional [dim 1: #samples, dim 2: channels, dims 3-4: kernel size]
        }
    
    #Training
    def update_swag(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        current_params = {name: param.detach() for name, param in self.network.named_parameters()} # MARI-PASCAL: why detach and not clone?

        # SWAG-diagonal
        for name, param in current_params.items():
            #MARI: update SWAG-diagonal attributes for weight `name` using `current_params` and `param`
            self.theta[name] = (self.n*self.theta[name] + param)/(self.n + 1)
            self.theta_squared[name] = (
                self.n*self.theta_squared[name] + param**2)/(self.n + 1)
           

        # Full SWAG
        if self.inference_mode == InferenceMode.SWAG_FULL:
            self._update_full_swag(current_params)

    def _update_full_swag(self, current_params) -> None:
        """Update the Full SWAG weight copies during training."""
        theta_difference = self._create_weight_copy()
        for name, param in current_params.items():
            theta_difference[name] = torch.clamp(param - self.theta[name], min=1e-10)
        self.weight_copies_full.append(theta_difference)

    def fit_swag(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag().
        """

        # Use SGD with momentum and weight decay to perform SWA.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        # Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_epochs,
            steps_per_epoch=len(loader),
        )

        # Perform initialization for SWAG fitting
        # PASCAL-TEAM-QUESTION: I think this might be "wrong" because the constructor
        # they added a note stating that we never have to work with all
        # the weights of the network at once but could only use the
        # weights of the current layer or something like that.
        # So we could maybe simplify this.
        self.theta = {name: param.detach().clone()
                      for name, param in self.network.named_parameters()}
        self.theta_squared = {name: param.detach().clone(
        )**2 for name, param in self.network.named_parameters()}


        self.network.train()
        with tqdm.trange(self.swag_epochs, desc="Running gradient descent for SWA") as pbar:
            pbar_dict = {}
            for epoch in pbar:
                average_loss = 0.0
                average_accuracy = 0.0
                num_samples_processed = 0
                for batch_xs, batch_is_snow, batch_is_cloud, batch_ys in loader:
                    optimizer.zero_grad()
                    pred_ys = self.network(batch_xs)
                    batch_loss = loss(input=pred_ys, target=batch_ys)
                    batch_loss.backward()
                    optimizer.step()
                    pbar_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    average_loss = (batch_xs.size(0) * batch_loss.item() + num_samples_processed * average_loss) / (
                        num_samples_processed + batch_xs.size(0)
                    )
                    average_accuracy = (
                        torch.sum(pred_ys.argmax(dim=-1) == batch_ys).item()
                        + num_samples_processed * average_accuracy
                    ) / (num_samples_processed + batch_xs.size(0))
                    num_samples_processed += batch_xs.size(0)
                    pbar_dict["avg. epoch loss"] = average_loss
                    pbar_dict["avg. epoch accuracy"] = average_accuracy
                    pbar.set_postfix(pbar_dict)

                # Implement periodic SWAG updates using the attributes defined in __init__
                if epoch % self.swag_update_freq == 0:
                    self.n = epoch // self.swag_update_freq + 1
                    self.update_swag()

# BASICALLY NOT DONE: CALIBRATE FUNCTION
    def calibrate(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate your predictions using a small validation set.
        validation_data contains well-defined and ambiguous samples,
        where you can identify the latter by having label -1.
        """
        if self.inference_mode == InferenceMode.MAP:
            # In MAP mode, simply predict argmax and do nothing else
            self._prediction_threshold = 0.0
            return

        # TODO(1): pick a prediction threshold, either constant or adaptive.
        #  The provided value should suffice to pass the easy baseline.
        self._prediction_threshold = 0.8  # 2.0 / 3.0



    #Training and/or inference ?
    def _update_batchnorm(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.train_dataset.
        They provide this method for convenience.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        old_momentum_parameters = dict()
        for module in self.network.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            old_momentum_parameters[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            module.reset_running_stats() #MARI what is this ???

        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.network.train()
        for (batch_xs,) in loader:
            self.network(batch_xs)
        self.network.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in old_momentum_parameters.items():
            module.momentum = momentum

# LEARNING RATE SCHEDULER
class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that calculates a different learning rate each gradient descent step.
    The default implementation keeps the original learning rate constant, i.e., does nothing.
    You can implement a custom schedule inside calculate_lr,
    and add+store additional attributes in __init__.
    You should not change any other parts of this class.
    """
    def calculate_lr(self, current_epoch: float, old_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        current_epoch is the fractional epoch of SWA fitting, starting at 0.
        That is, an integer value x indicates the start of epoch (x+1),
        and non-integer values x.y correspond to steps in between epochs (x+1) and (x+2).
        old_lr is the previous learning rate.

        This method should return a single float: the new learning rate.
        """
        return old_lr

# DETAILS - TO IMPROVE
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch /
                              self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]
