import enum
import torch
import typing
import math
import warnings
import collections
import tqdm


class SWAGInference(object):

    def __init__(
        self,
        train_xs: torch.Tensor,
        inference_mode: 'swag-diagonal',

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
        #MARI TODO: ?????self.weight_copies_full.clear()
        self.weight_copies_full = collections.deque(maxlen=swag_epochs * swag_update_freq) #Pascal's team did different
        # MARI: This deque will store multiple weight copies over the course of training
        # MARI-PASCAL: possibly use something different instead of deque [computationally more efficient but 'clear' may be better]

        #MARI-RIC TODO:
        # Calibration, prediction, and other attributes
        # TODO(2): create additional attributes, e.g., for calibration
        
        self._prediction_threshold = None  # this is an example, feel free to be creative #RIC: find threshold
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
        # MARI-PASCAL: why clone and not detach? to make an actual copy, not a ref, to have cleaner stuff
        current_params = {name: param.clone() for name, param in self.network.named_parameters()} 

        # SWAG-diagonal
        for name, param in current_params.items():
            #MARI: update SWAG-diagonal attributes for weight `name` using `current_params` and `param`
            self.theta[name] = (self.n*self.theta[name] + param)/(self.n + 1)
            self.theta_squared[name] = (
                self.n*self.theta_squared[name] + param**2)/(self.n + 1)
           

        # Full SWAG
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
        # Will be replaced with Pascal's stuff (use same as pretrained, i.e. Adam + binary-focal-loss on 1-to-many class)

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

        # Will NOT be replaced with Pascal's stuff (use same as pretrained, i.e. Adam + binary-focal-loss)

        # Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_epochs,
            steps_per_epoch=len(loader),
        )

        # Perform initialization for SWAG fitting

        # Mari's interpretation: all the computations done are per layer 
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
                for batch_xs, batch_ys in loader:
                    optimizer.zero_grad()
                    pred_ys = self.network(batch_xs)
                    batch_loss = loss(input=pred_ys, target=batch_ys)
                    batch_loss.backward()
                    optimizer.step()
                    pbar_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()
                    
                    
                    # Most of this :point_down: is updating metrics
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
                # End of just updating metrics

                # Implement periodic SWAG updates using the attributes defined in __init__
                if epoch % self.swag_update_freq == 0:
                    self.n = epoch // self.swag_update_freq + 1
                    self.update_swag()

# NOT DONE YET: CALIBRATE FUNCTION - MARI-PASCAL-RIC
    def calibrate(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate predictions using a small validation set.
        """
        # TODO MARI-PASCAL-RIC: implement threshold that makes sense
        # TODO either use validation_data as training set for temperature scaling? if not needed, kill it
        self._prediction_threshold = 0.8  # 2.0 / 3.0

# TODO: once CALIBRATE is done, reuse a sort of scaling factor output of calibration, and this'll be the difference between 
    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        Outputs should be a Nx6 tensor, where N is the number of samples in loader,
        and all rows of the output should sum to 1.
        That is, output row i column j should be your predicted p(y=j | x_i).
        """

        #Telling pytorch not to update gradients pls - read documentation but it's a "locker" functionality
        self.network.eval()

        # Perform Bayesian model averaging:
        # Instead of sampling self.bma_samples networks (using self.sample_parameters())
        # for each datapoint, you can save time by sampling self.bma_samples networks,
        # and perform inference with each network on all samples in loader.
        per_model_sample_predictions = []
        for _ in tqdm.trange(self.bma_samples, desc="Performing Bayesian model averaging"):
            self.sample_parameters()  # Samples new weights and loads it into our DNN

            # ATTENTION: per_model_sample_predictions contints a list of all predictions for all models.
            # i.e. per_model_sample_predictions = [ predictions_of_model_1, predictions_of_model_2, ...]

            # predictions is the predictions of one model for all samples in loader


            predictions = self.predict_probabilities_vanilla(loader)
            per_model_sample_predictions.append(predictions)

        assert len(per_model_sample_predictions) == self.bma_samples
        assert all(
            isinstance(model_sample_predictions, torch.Tensor)
            and model_sample_predictions.dim() == 2  # N x C
            and model_sample_predictions.size(1) == 6
            for model_sample_predictions in per_model_sample_predictions
        )

        # Add all model predictions together and take the mean
        bma_probabilities = torch.mean(torch.stack(
            per_model_sample_predictions, dim=0), dim=0)

        assert bma_probabilities.dim() == 2 and bma_probabilities.size(1) == 6  # N x C
        return bma_probabilities

    def predict_probabilities_vanilla(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        This function does the same as old predict_probabilities_map that you can use as placeholder
        for the logic until we have calibration and proper SWAG-prediction functionality
        """
        predictions = []
        for (batch_xs,) in loader:
            predictions.append(self.network(batch_xs))

        predictions = torch.cat(predictions)
        return torch.softmax(predictions, dim=-1)
    

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        Hence, after calling this method, self.network corresponds to a new posterior sample.
        """

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.

        for name, param in self.network.named_parameters():
            # SWAG-diagonal part
            z_1 = torch.randn(param.size())

            # Sample parameter values for SWAG-diagonal
            # raise NotImplementedError("Sample parameter for SWAG-diagonal")
            current_mean = self.theta[name]
            current_std = torch.sqrt(torch.clamp(
                self.theta_squared[name] - current_mean**2, min=1e-10))

            assert current_mean.size() == param.size() and current_std.size() == param.size()

            # Diagonal part
            sampled_param = current_mean + (1.0/math.sqrt(2.0))*current_std*z_1

            # TODO MARI-PASCAL-RIC: review formula
            # Here we are doing the multiplication layer-by-layer using the same z_2 every time
            # so a sort of periodic randomness which may be wrong -> with full-swag solution, may improve "randomness"
            # PASCAL TEAM : unsure about how they didi the multiplication `) * torch` bit
            # Full SWAG part
            z_2 = torch.randn(param.size()) #unsure of the logic here and flexibility of formula

            for D_i in self.weight_copies_full:
                sampled_param += (1.0 / math.sqrt(2 * self.deviation_matrix_max_rank - 1)
                                  ) * torch.clamp(D_i[name]*z_2, min=1e-10)

            # Modify weight value in-place; directly changing self.network
            param.data = sampled_param

        #don't remember much why, but bc we update the weights in place, we re-do batch-normalization at the end of this fn
        #shouldn't be a big deal so no worries (f.ex. it may just be "scaling the weights" or "normalizing distrib")
        self._update_batchnorm()



    #Training and/or inference ? TODO: read documentation about it :nerd_face:
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

    def calculate_lr(self, current_epoch: float, old_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        current_epoch is the fractional epoch of SWA fitting, starting at 0.
        That is, an integer value x indicates the start of epoch (x+1),
        and non-integer values x.y correspond to steps in between epochs (x+1) and (x+2).
        old_lr is the previous learning rate.

        This method should return a single float: the new learning rate.
        """
        # MARI TODO: can define a list of lr-s and cycle through
        # NOTE: it's possible that in SWA paper they try constant and cyclic, but that in the SWAG, only constant
        # so perhaps we don't mind much - it's a possible improvement
        return old_lr


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
