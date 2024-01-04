class SWAGInference:
    def __init__(self) -> None:
        # TODO: Fix randomness

        # TODO: Defined num swag epochs
        # TODO: Define swag learning rate
        # TODO: Define swag update frequency
        # TODO: Define deviation matrix max rank
        # TODO: Define number of BMA samples

        # TODO: Load training data
        # TODO: Load test data
        # TODO: Load model
        # TODO: Load optimizer
        # TODO: Define loss
        # TODO: Define learning rate scheduler

        # TODO: Allocate memory for theta
        # TODO: Allocate memory for theta_squared
        # TODO: Allocate memory for D
        
        # TODO: Define prediction thresholt (used for calibration I think)
        # TODO: set current iteration n - used to compute sigma etc.
        pass

    def update_swag(self) -> None:
        # TODO: Compute SWAG-Diagonal
        # TODO: Compute Full SWAG
        pass

    def fit_swag(self) -> None:
        # TODO: init theta and theta_squared
        # TODO: set network in training mode
        # TODO: run swag epochs amount of epochs
        #       for each epoch (resp update freq.) run self.update_swag()
        pass

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
        # TODO: Helper method used to allocate memory for weights.
        pass

    def _update_batchnorm(self) -> None:
        # TODO: Helper method used to update batchnorm.
        # Understand why we actually have to do that
        pass