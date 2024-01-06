import torch
import collections
import tqdm
import math
from utils.load_model import load_model


class SWAGInference:
    def __init__(self,
                 X_train, #beginning: for dilatedCNN
                 y_train,
                 X_val,
                 y_val,
                 X_test,
                 y_test,
                 CV_k,
                 checkpoint_path=None, #end: for dilatedCNN
                 swag_epochs=30,
                 swag_update_freq=1,
                 swag_learning_rate=0.045,
                 deviation_matrix_max_rank=15,
                 bma_samples=30) -> None:

        #Fix randomness
        torch.manual_seed(42)

        # Define num swag epochs, swag LR, swag update freq, deviation matrix max rank, num BMA samples
        self.swag_epochs = swag_epochs
        self.swag_learning_rate = swag_learning_rate
        self.swag_update_freq = swag_update_freq
        self.deviation_matrix_max_rank = deviation_matrix_max_rank
        self.bma_samples = bma_samples

        #TODO: Network - DOES THIS MAKE SENSE?
        self.network = load_model(
            checkpoint_path,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            CV_k,
        )

        # Load training data, test data
        # TODO: test and train as in CNN split ???
        self.train_loader = self.network.train_loader
        self.test_loader = self.network.test_loader


        # Load model
        self.model = self.network.model

        # TODO: understand what to load when Load optimizer - network and swag_learning_rate
        # Define optimizer and learning rate scheduler
        self.optimizer =  self.network.optimizer

        cycle_len = 5  # You can adjust the cycle length
        # NOTE: could add a step_size_down parameter (set it to cycle_len) so the cyclic pattern is not just increasing
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=5, gamma=1 #testing variables – no strings attached
            )

        ### NOTE: or try 
        # torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=swag_learning_rate / 10,
        #                                   max_lr=swag_learning_rate, step_size_up=cycle_len, cycle_momentum=True)


        # Define loss
        self.loss_fn = self.network.loss_fn

        # Allocate memory for theta, theta_squared, D (D-matrix)
        self.theta = self._create_weight_copy()
        self.theta_squared = self._create_weight_copy()
        self.D = self._create_weight_copy()

        # ADDED: for full SWAG
        self.weight_copies =  collections.deque(maxlen=swag_epochs * swag_update_freq) #Pascal's team did different
        
        # Define prediction threshold (used for calibration I think)
        self.prediction_threshold = 0.5 # TODO: Ric help think of a sensible threshold

        # set current iteration n - used to compute sigma etc.
        self.n = 0


    def update_swag(self) -> None:
        # TODO: Check that this works (esp. the current_params thing)
        current_params = {name: param.detach().clone() for name, param in self.model.named_parameters()} 

        # Update swag diagonal
        for name, param in current_params:
            print(f"Updating {name} with param {param}")
            self.theta[name] = (self.n*self.theta[name] + param)/(self.n + 1)
            self.theta_squared[name] = (
                self.n*self.theta_squared[name] + param**2)/(self.n + 1)
            
        #Update full swag
        theta_difference = self._create_weight_copy()
        for name, param in current_params.items():
            theta_difference[name] = torch.clamp(param - self.theta[name], min=1e-10)
        self.weight_copies.append(theta_difference)

    def fit_swag(self) -> None:
        # TODO: MARI review the whole logic in details and test
        # What this should do: init theta and theta_squared
        # set network in training mode
        # run swag epochs amount of epochs
        # for each epoch (resp update freq.) run self.update_swag()

        # TODO: review these variables defs
        loader = self.train_loader
        loss = self.loss_fn
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler

        self.theta = {name: param.detach().clone()
                      for name, param in self.model.named_parameters()}
        self.theta_squared = {name: param.detach().clone(
        )**2 for name, param in self.model.named_parameters()}

        self.model.train()
        with tqdm.trange(self.swag_epochs, desc="Running gradient descent for SWA") as pbar:
            pbar_dict = {}
            for epoch in pbar:
                average_loss = 0.0
                average_accuracy = 0.0
                num_samples_processed = 0
                print("\n about to start for loop on loader \n")
                # items in loader are tuples: first elt is a list of strings like 'A0194m', 
                # the second element is a tuple of two tensors.
                # The tensors correspond to your input data (batch_xs) and target labels (batch_ys)
                for item in loader:
                    names = item[0]
                    batch_data = item[1]  # batch_data.shape: torch.Size([110, 12, 5000])

                    # If batch_data is a tensor containing both input data and target labels, you might need to further split it
                    batch_xs = batch_data[:, :-1, :]  # Assuming the last dimension represents features
                    batch_ys = batch_data[:, -1, :]   # Assuming the last dimension represents labels
                    print(f"First name of loader item: {names[0]}")
                    print(f"batch_data.shape: {batch_data.shape}")

                    optimizer.zero_grad()
                    pred_ys = self.network(batch_xs)
                    #BinaryFocalLoss takes as input prediction_logits, labels, self.model.training according to dilatedCNN.py implementation
                    batch_loss = loss(input=pred_ys, target=batch_ys)
                    batch_loss.backward() # MARI: FIX THIS
                    optimizer.step()
                    pbar_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()
                    
                    
                    # Most of this :point_down: is updating metrics
                    # Calculate cumulative average training loss and accuracy
                    #NOTE: batch_loss.item() should be replaced if working in a multi-GPU environment (?) (use torch.Tensor.cpu().numpy())
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

    def calibrate(self) -> None:
        # TODO: Calibrate model. Ask Ric. For beginning we can just
        # set the prediction threshold to some value.
        # Maybe calibration is done with temperature scaling?
        pass

    def predict_probabilities(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Goal: Implement Bayesian Model Averaging by doing:
             1. Sample new model from SWAG (call self.sample_parameters())
             2. Predict probabilities with sampled model
             3. repeat 1-2 for num_bma_samples times
             4. Average the probabilities over the num_bma_samples
        """
        # QUESTION: should `loader` be an input so one cass pass in `self.test_loader` vs `self.train_loader` ?

        with torch.no_grad():
            self.model.eval()

            # Perform Bayesian model averaging:
            # Instead of sampling self.bma_samples networks (using self.sample_parameters())
            # for each datapoint, you can save time by sampling self.bma_samples networks,
            # and perform inference with each network on all samples in loader.


            # per_model_sample_predictions contints a list of all predictions for all models.
            # i.e. per_model_sample_predictions = [ predictions_of_model_1, predictions_of_model_2, ...]
            per_model_sample_predictions = []
            for _ in tqdm.trange(self.bma_samples, desc="Performing Bayesian model averaging"):
                self.sample_parameters()  # Samples new weights and loads it into our DNN

                # predictions is the predictions of one model for all samples in loader

                predictions = self._predict_probabilities_vanilla(loader)
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

    def sample_parameters(self) -> None:
        # Note: Be sure to do the math here correctly!
        # The solution might have a tiny error in it.

        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.

        # Loop over each layer in the model (self.model.named_parameters())
        for name, param in self.model.named_parameters():
            # SWAG-diagonal part
            # Draw vectors z_1 and z_2 almost randomly
            z_1 = torch.normal(mean=0.0, std=1.0, size=param.size()) # random sample diagonal
            z_2 = torch.normal(mean=0.0, std=1.0, size=param.size()) # random sample full SWAG

            # Compute mean and std
            current_mean = self.theta[name]
            current_std = torch.normal(mean=current_mean, std=torch.ones_like(current_mean))

            assert current_mean.size() == param.size() and current_std.size() == param.size()

            ## Diagonal part

            # Compute diagonal covariance matrix using mean + std * z_1
            sampled_param = current_mean + (1.0 / math.sqrt(2.0)) * current_std * z_1

            ## Full SWAG part

            # Compute full covariance matrix by doing D * z_2
            sampled_param += (1.0 / math.sqrt(2 * self.deviation_matrix_max_rank - 1)) * torch.sum(
                D_i[name] * z_2 for D_i in self.weight_copies
            )

            # Load sampled params into model
            # Modify weight value in-place; directly changing self.network
            param.data = sampled_param


        # Update batchnorm
        self._update_batchnorm()

    def predict_labels(self) -> None:
        # TODO: Predict labels. One needs to do much rsearch for this
        # because now we can return "I don't know" but do we really
        # want to do that? The challenge score doesn't respect an
        # "I don't know" label. This could be some extra research
        # though and we can compute the score for the case where
        # we ignore those "I don't know" cases.
        pass

    def _create_weight_copy(self) -> None:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight (matrix)"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.model.named_parameters()
        }

    def _predict_probabilities_vanilla(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        # TODO: Use more sophisticated prediction method than softmax
        predictions = []
        for (batch_xs,) in loader:
            predictions.append(self.network(batch_xs))

        predictions = torch.cat(predictions)
        return torch.softmax(predictions, dim=-1)


    def _update_batchnorm(self) -> None:
        # TODO: Helper method used to update batchnorm.
        # Understand why we actually have to do that
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
        for module in self.model.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            old_momentum_parameters[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            #  :eyes: reset the running statistics of the batch normalization layer,
            #  including mean and variance. Done to compute fresh statistics
            #  using the entire training dataset during the inference phase
            module.reset_running_stats()

        loader = torch.utils.data.DataLoader(
            self.train_loader.dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.model.train()
        for (batch_xs,) in loader:
            self.network(batch_xs)
        self.model.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in old_momentum_parameters.items():
            module.momentum = momentum

## TODO:
            # 1. Decide whether to keep self.network.model... or self.model...