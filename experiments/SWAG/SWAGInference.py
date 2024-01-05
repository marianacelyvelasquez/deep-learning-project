import torch
import collections
import tqdm
import torch.nn as nn
from experiments.dilated_CNN.dilatedCNN import dilatedCNNExperiment


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
        self.network = dilatedCNNExperiment(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            CV_k,
            checkpoint_path=checkpoint_path,
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
        self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=swag_learning_rate / 10,
                                                         max_lr=swag_learning_rate, step_size_up=cycle_len)

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
        current_params = {name: param.detach().clone() for name, param in self.network.model.named_parameters()} 

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
                      for name, param in self.network.named_parameters()}
        self.theta_squared = {name: param.detach().clone(
        )**2 for name, param in self.network.named_parameters()}

        self.network.model.train()
        with tqdm.trange(self.swag_epochs, desc="Running gradient descent for SWA") as pbar:
            pbar_dict = {}
            for epoch in pbar:
                average_loss = 0.0
                average_accuracy = 0.0
                num_samples_processed = 0
                for batch_xs, batch_ys in loader:
                    optimizer.zero_grad()
                    pred_ys = self.network(batch_xs)
                    #BinaryFocalLoss takes as input prediction_logits, labels, self.network.model.training according to dilatedCNN.py implementation
                    batch_loss = loss(input=pred_ys, target=batch_ys)
                    batch_loss.backward()
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

    def predict_probabilities(self) -> None:
        # TODO: Set model into eval mode
        # TODO: Implement Bayesian Model Averaging by doing:
        #      1. Sample new model from SWAG (call self.sample_parameters())
        #      2. Predict probabilities with sampled model
        #      3. repeat 1-2 for num_bma_samples times
        #      4. Average the probabilities over the num_bma_samples
        pass

    def sample_parameters(self) -> None:
        # TODO: Loop over each layer in the model (self.model.named_parameters())
        # TODO: Draw a vector z_1 and z_2 randomly
        # TODO: Compute mean and std
        # TODO: Compute diagonal covariance matrix using mean + std*z_1
        # TODO: Compute full covariance matrix by doing D*z_2
        # TODO: Load sampled params into model
        # TODO: Update batchnorm

        # Note: Be sure to do the math here correctly!
        # The solution might have a tiny error in it.
        pass

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
            for name, param in self.network.named_parameters()
        }

    def _update_batchnorm(self) -> None:
        # TODO: Helper method used to update batchnorm.
        # Understand why we actually have to do that
        pass