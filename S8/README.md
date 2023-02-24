# Code Structure
1. File S8.ipynb is the colab notebook
4. Folder plots has photoes with random image examples as well as mis classified examples.


--------------------------------------------------------------------------------------------------------------------------------

# Example Images

![training_images](https://user-images.githubusercontent.com/21367838/221280719-d58a8fa9-93a2-4261-96fe-b80f633279ec.png)


--------------------------------------------------------------------------------------------------------------------------------

# Final Model Architecture - Uses traditional convolution

1. Prep layer: 1 convolution layer of kernel size 3 x 3, follwed by Batchnorm & ReLu. (64 kernels)
2. Custom Residual Block: Convolution on output from last layer. Residual Block on output from last layer. Add output from both steps. (128 kernels)
3. Convolution layer: 1 convolution layer of kernel size 3 x 3, follwed by MaxPool, Batchnorm & ReLu. (256 kernels)
4. Custom Residual Block: Convolution on output from last layer. Residual Block on output from last layer. Add output from both steps. (512 kernels)
5. Max pooling of kernel size 4
6. Fully connected layer


![image](https://user-images.githubusercontent.com/21367838/221282435-d804c655-6f08-40b6-bc06-96245c99a512.png)


--------------------------------------------------------------------------------------------------------------------------------

# Model Training Strategy
1. Model wa strained for 24 epochs using batch size of 512 using cross entropy loss and sgd optimizer with default parameters.
2. One cycle policy was used for sceduling learning rate..
3. Data augmentation stragey was applied and padding, random crop and coarse dropout method was used.


--------------------------------------------------------------------------------------------------------------------------------

# Training Logs

EPOCH: 0
Loss = 1.18 | Batch = 97 | Accuracy = 42.31: 100% 98/98 [00:24<00:00,  3.97it/s]

Test set: Average loss: 0.00251, Accuracy: 55.92

EPOCH: 1
Loss = 1.15 | Batch = 97 | Accuracy = 60.60: 100% 98/98 [00:24<00:00,  3.95it/s]

Test set: Average loss: 0.00206, Accuracy: 63.66

EPOCH: 2
Loss = 0.76 | Batch = 97 | Accuracy = 66.61: 100% 98/98 [00:25<00:00,  3.85it/s]

Test set: Average loss: 0.00176, Accuracy: 69.28

EPOCH: 3
Loss = 0.76 | Batch = 97 | Accuracy = 71.01: 100% 98/98 [00:26<00:00,  3.76it/s]

Test set: Average loss: 0.00145, Accuracy: 74.21

EPOCH: 4
Loss = 0.69 | Batch = 97 | Accuracy = 74.85: 100% 98/98 [00:27<00:00,  3.58it/s]

Test set: Average loss: 0.00161, Accuracy: 72.41

EPOCH: 5
Loss = 0.52 | Batch = 97 | Accuracy = 77.87: 100% 98/98 [00:25<00:00,  3.83it/s]

Test set: Average loss: 0.00126, Accuracy: 78.14

EPOCH: 6
Loss = 0.65 | Batch = 97 | Accuracy = 80.25: 100% 98/98 [00:25<00:00,  3.87it/s]

Test set: Average loss: 0.00117, Accuracy: 79.51

EPOCH: 7
Loss = 0.63 | Batch = 97 | Accuracy = 82.03: 100% 98/98 [00:26<00:00,  3.65it/s]

Test set: Average loss: 0.00108, Accuracy: 81.49

EPOCH: 8
Loss = 0.42 | Batch = 97 | Accuracy = 83.62: 100% 98/98 [00:26<00:00,  3.73it/s]

Test set: Average loss: 0.00100, Accuracy: 83.09

EPOCH: 9
Loss = 0.48 | Batch = 97 | Accuracy = 84.87: 100% 98/98 [00:26<00:00,  3.72it/s]

Test set: Average loss: 0.00098, Accuracy: 83.38

EPOCH: 10
Loss = 0.36 | Batch = 97 | Accuracy = 86.24: 100% 98/98 [00:26<00:00,  3.76it/s]

Test set: Average loss: 0.00107, Accuracy: 82.15

EPOCH: 11
Loss = 0.41 | Batch = 97 | Accuracy = 87.03: 100% 98/98 [00:25<00:00,  3.89it/s]

Test set: Average loss: 0.00120, Accuracy: 80.54

EPOCH: 12
Loss = 0.35 | Batch = 97 | Accuracy = 87.71: 100% 98/98 [00:25<00:00,  3.91it/s]

Test set: Average loss: 0.00091, Accuracy: 84.76

EPOCH: 13
Loss = 0.30 | Batch = 97 | Accuracy = 88.78: 100% 98/98 [00:25<00:00,  3.78it/s]

Test set: Average loss: 0.00087, Accuracy: 85.73

EPOCH: 14
Loss = 0.35 | Batch = 97 | Accuracy = 89.43: 100% 98/98 [00:26<00:00,  3.73it/s]

Test set: Average loss: 0.00083, Accuracy: 85.83

EPOCH: 15
Loss = 0.35 | Batch = 97 | Accuracy = 90.34: 100% 98/98 [00:25<00:00,  3.80it/s]

Test set: Average loss: 0.00079, Accuracy: 86.87

EPOCH: 16
Loss = 0.26 | Batch = 97 | Accuracy = 90.95: 100% 98/98 [00:24<00:00,  3.96it/s]

Test set: Average loss: 0.00072, Accuracy: 87.94

EPOCH: 17
Loss = 0.26 | Batch = 97 | Accuracy = 91.72: 100% 98/98 [00:25<00:00,  3.87it/s]

Test set: Average loss: 0.00072, Accuracy: 88.14

EPOCH: 18
Loss = 0.20 | Batch = 97 | Accuracy = 92.43: 100% 98/98 [00:26<00:00,  3.75it/s]

Test set: Average loss: 0.00068, Accuracy: 88.60

EPOCH: 19
Loss = 0.26 | Batch = 97 | Accuracy = 92.88: 100% 98/98 [00:27<00:00,  3.59it/s]

Test set: Average loss: 0.00074, Accuracy: 87.72

EPOCH: 20
Loss = 0.18 | Batch = 97 | Accuracy = 93.55: 100% 98/98 [00:25<00:00,  3.89it/s]

Test set: Average loss: 0.00064, Accuracy: 88.82

EPOCH: 21
Loss = 0.15 | Batch = 97 | Accuracy = 94.16: 100% 98/98 [00:25<00:00,  3.90it/s]

Test set: Average loss: 0.00064, Accuracy: 89.48

EPOCH: 22
Loss = 0.13 | Batch = 97 | Accuracy = 94.81: 100% 98/98 [00:25<00:00,  3.84it/s]

Test set: Average loss: 0.00059, Accuracy: 90.06

EPOCH: 23
Loss = 0.13 | Batch = 97 | Accuracy = 95.32: 100% 98/98 [00:25<00:00,  3.78it/s]

Test set: Average loss: 0.00058, Accuracy: 90.15


--------------------------------------------------------------------------------------------------------------------------------

# Final Model Performance

Train set: Average loss: 0.00028, Accuracy: 95.79 \
Test set: Average loss: 0.00058, Accuracy: 90.15


--------------------------------------------------------------------------------------------------------------------------------

# Learning Rate Schedule

![one_cycle_lr_plot](https://user-images.githubusercontent.com/21367838/221283087-e2670d30-3c3c-455b-a6ee-e66cda290d48.png)


--------------------------------------------------------------------------------------------------------------------------------

# Mis-classified Images

![misclassified_images](https://user-images.githubusercontent.com/21367838/221282941-72361ad7-9479-4f7a-978e-8579945aa73a.png)


--------------------------------------------------------------------------------------------------------------------------------

# Performance Plots

![performance_plot](https://user-images.githubusercontent.com/21367838/221283196-3838fa81-50e4-4421-9e96-72a61b68b819.png)
