import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
import threading
import multiprocessing
import numpy as np
import queue

# Step 1: Define a Simple Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example architecture

    def forward(self, x):
        x = self.fc(x)
        return x

# Function to train the model
def train_model(model, device, train_loader, criterion, optimizer):
    model.to(device)
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

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

# Step 3: Data Preparation
# Assume X, y are your data and labels
X = np.random.randn(100, 10)  # Example data
y = np.random.randn(100, 1)   # Example labels

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 10-fold CV
kf = KFold(n_splits=10)

# Task Queue
task_queue = queue.Queue()

# Define a Worker Thread
def worker(device_id):
    while not task_queue.empty():
        try:
            fold, train_index, test_index = task_queue.get_nowait()
        except queue.Empty:
            break

        train_data = torch.utils.data.TensorDataset(X_tensor[train_index], y_tensor[train_index])
        test_data = torch.utils.data.TensorDataset(X_tensor[test_index], y_tensor[test_index])

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=10)

        model = SimpleModel()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        device = torch.device(f'{device_type}:{device_id}')
        train_model(model, device, train_loader, criterion, optimizer)
        # Add testing and evaluation logic if needed

        task_queue.task_done()

# Populate the Task Queue
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    task_queue.put((fold, train_index, test_index))

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
print("Training complete!")
