# REMEMBER TO ADD:
import multiprocessing

def predict_probabilities(self, loader: torch.utils.data.DataLoader): # NOTE: this function is currently never called -  rewrite based on dilatedCNN, do stacking and appending predictions [implement BMA yourself based on CNN code]
        """
        Goal: Implement Bayesian Model Averaging by doing:
             1. Sample new model from SWAG (call self.sample_parameters())
             2. Predict probabilities with sampled model
             3. repeat 1-2 for num_bma_samples times
             4. Average the probabilities over the num_bma_samples
        """
        # :bulb: we have all the weights and the parameters obtained by the SWAG training, let's actually do inference

        with torch.no_grad():
            """
            Evaluate (self.bma_samples different) models on test data; Collect them; Get the mean of it (BMA)
            """
            self.model.eval()

            # Perform Bayesian model averaging:
            # Instead of sampling self.bma_samples networks (using self.sample_parameters())
            # for each datapoint, you can save time by sampling self.bma_samples networks,
            # and perform inference with each network on all samples in loader.
            
            # How many CPU are AVAILABLE to us
            num_av_cpus = len(os.sched_getaffinity(0))
            print(f"for fun: {num_av_cpus}")

            # Device Selection
            def get_best_device():
                if torch.cuda.is_available():
                    return 'cuda', torch.cuda.device_count()
                else:
                    return 'cpu', num_av_cpus
                
            device_type, device_count = get_best_device()

            # Task Queue
            task_queue = queue.Queue()

            # per_model_sample_predictions contains a list of all predictions for all models.
            # i.e. per_model_sample_predictions = [ predictions_of_model_1, predictions_of_model_2, ...]
            per_model_sample_predictions = []

            # Define a Worker Thread
            def worker(device_id):
                while not task_queue.empty():
                    try:
                        _ = task_queue.get_nowait()
                    except queue.Empty:
                        break

                    # will do 1. and 2. of BMA in this "loop"
                    self.sample_parameters()  # Samples new weights and loads it into our DNN

                    # predictions is the predictions of one model for all samples in loader
                    predictions = self._predict_probabilities_of_model(loader) # Here use CNN+sigmoid obtained probabilities

                    per_model_sample_predictions.append(predictions) # HERE: 

                    ## TODO: better assertions
                    # assert len(per_model_sample_predictions) == self.bma_samples
                    # assert all(
                    #     isinstance(model_sample_predictions, torch.Tensor)
                    #     and model_sample_predictions.dim() == 2  # N x C # I SHOULD GET AN ERROR HERE -> 3 (?)
                    #     and model_sample_predictions.size(1) == 6 # I SHOULD GET AN ERROR HERE -> 24
                    #     for model_sample_predictions in per_model_sample_predictions
                    # )

                    task_queue.task_done()

            # Populate the Task Queue
            for _ in range(self.bma_samples):
                task_queue.put(_)

            # Create and start threads
            threads = []
            for i in range(device_count):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            task_queue.join()

            # Add all model predictions together and take the mean
            bma_probabilities = torch.mean(torch.stack(
                per_model_sample_predictions, dim=0), dim=0) # this is the difference between swag and cnndilated

            #assert bma_probabilities.dim() == 2 and bma_probabilities.size(1) == 6  # N x C
            return bma_probabilities