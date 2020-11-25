import numpy as np
import torch

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

# Converting to Tensors

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Defining the weights and biases

w = torch.randn(2, 3, requires_grad = True)
b = torch.randn(2, requires_grad = True)

# Defining the model

def model(x):
  return x @ w.t() + b

preds = model(inputs)
loss = mse(preds, target)

print('Loss before optimization: ', loss)

# For a step backwards:

#loss.backward()

# To get gradient of loss

#w.grad   #gradient w.r.t w
#b.grad   #gradient w.r.t b

# Defining the loss function

# MSE Loss:

def mse(t1, t2):
  diff = t1 - t2
  return torch.sum(diff*diff)/diff.numel()

for i in range(100):
  preds = model(inputs)
  loss = mse(preds, targets)
  loss.backward()
  with torch.no_grad():
    w -= w.grad + 1e-5
    b -= b.grad + 1e-5
    w.grad.zero_()
    b.grad.zero_()

preds = model(inputs)
loss = mse(preds, targets)
print('Loss after optimization: ', loss)