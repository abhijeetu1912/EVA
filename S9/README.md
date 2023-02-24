# Code Structure
1. File S9.ipynb is the colab notebook
4. Folder plots has photoes with random image examples as well as mis classified examples.


--------------------------------------------------------------------------------------------------------------------------------

# Example Images

![training_images](https://user-images.githubusercontent.com/21367838/221285882-f2205fa3-adfc-432b-a006-e5b746bbbbb4.png)


--------------------------------------------------------------------------------------------------------------------------------

# Final Model Architecture - Uses traditional convolution

1. 3 consecutive convolution layers with 16, 32 and 48 kernels of size 3x3 respectively.
2. 4 ultimus blocks where 1. creat key, 2. create query, 3. create value, 4. softmax(query transpose * key)/(8^0.5), 5. multiply it with value. All multiplications are matrix multiplications.
3. Fully connected layer.


![image](https://user-images.githubusercontent.com/21367838/221282435-d804c655-6f08-40b6-bc06-96245c99a512.png)


--------------------------------------------------------------------------------------------------------------------------------

# Model Training Strategy
1. Model wa strained for 24 epochs using batch size of 256 using cross entropy loss and sgd optimizer with default parameters.
2. One cycle policy was used for sceduling learning rate..
3. Data augmentation stragey was applied and padding, random crop and coarse dropout method was used.


--------------------------------------------------------------------------------------------------------------------------------

# Training Logs

EPOCH: 0
Loss = 1.98 | Batch = 195 | Accuracy = 15.22: 100% 196/196 [00:13<00:00, 14.16it/s]

Test set: Average loss: 0.00779, Accuracy: 20.69

EPOCH: 1
Loss = 1.76 | Batch = 195 | Accuracy = 21.45: 100% 196/196 [00:14<00:00, 13.80it/s]

Test set: Average loss: 0.00750, Accuracy: 22.65

EPOCH: 2
Loss = 2.10 | Batch = 195 | Accuracy = 17.42: 100% 196/196 [00:14<00:00, 14.00it/s]

Test set: Average loss: 0.01069, Accuracy: 17.81

EPOCH: 3
Loss = 1.95 | Batch = 195 | Accuracy = 20.81: 100% 196/196 [00:13<00:00, 14.07it/s]

Test set: Average loss: 0.00810, Accuracy: 18.74

EPOCH: 4
Loss = 1.83 | Batch = 195 | Accuracy = 22.05: 100% 196/196 [00:13<00:00, 14.04it/s]

Test set: Average loss: 0.00760, Accuracy: 22.16

EPOCH: 5
Loss = 1.95 | Batch = 195 | Accuracy = 23.04: 100% 196/196 [00:14<00:00, 13.87it/s]

Test set: Average loss: 0.00759, Accuracy: 21.40

EPOCH: 6
Loss = 1.85 | Batch = 195 | Accuracy = 23.96: 100% 196/196 [00:13<00:00, 14.13it/s]

Test set: Average loss: 0.00851, Accuracy: 18.21

EPOCH: 7
Loss = 1.94 | Batch = 195 | Accuracy = 19.76: 100% 196/196 [00:16<00:00, 12.13it/s]

Test set: Average loss: 0.00765, Accuracy: 21.32

EPOCH: 8
Loss = 1.86 | Batch = 195 | Accuracy = 22.59: 100% 196/196 [00:14<00:00, 13.93it/s]

Test set: Average loss: 0.00741, Accuracy: 23.98

EPOCH: 9
Loss = 1.80 | Batch = 195 | Accuracy = 23.85: 100% 196/196 [00:14<00:00, 13.64it/s]

Test set: Average loss: 0.00782, Accuracy: 22.15

EPOCH: 10
Loss = 1.77 | Batch = 195 | Accuracy = 25.06: 100% 196/196 [00:14<00:00, 13.32it/s]

Test set: Average loss: 0.00854, Accuracy: 20.62

EPOCH: 11
Loss = 1.75 | Batch = 195 | Accuracy = 26.51: 100% 196/196 [00:13<00:00, 14.08it/s]

Test set: Average loss: 0.00711, Accuracy: 27.24

EPOCH: 12
Loss = 1.70 | Batch = 195 | Accuracy = 26.95: 100% 196/196 [00:14<00:00, 14.00it/s]

Test set: Average loss: 0.00717, Accuracy: 26.87

EPOCH: 13
Loss = 1.77 | Batch = 195 | Accuracy = 27.62: 100% 196/196 [00:14<00:00, 13.94it/s]

Test set: Average loss: 0.00719, Accuracy: 27.00

EPOCH: 14
Loss = 1.84 | Batch = 195 | Accuracy = 27.79: 100% 196/196 [00:14<00:00, 13.77it/s]

Test set: Average loss: 0.00710, Accuracy: 28.27

EPOCH: 15
Loss = 1.73 | Batch = 195 | Accuracy = 28.34: 100% 196/196 [00:14<00:00, 13.84it/s]

Test set: Average loss: 0.00899, Accuracy: 19.82

EPOCH: 16
Loss = 1.70 | Batch = 195 | Accuracy = 28.75: 100% 196/196 [00:14<00:00, 13.90it/s]

Test set: Average loss: 0.00710, Accuracy: 28.87

EPOCH: 17
Loss = 1.69 | Batch = 195 | Accuracy = 28.98: 100% 196/196 [00:13<00:00, 14.21it/s]

Test set: Average loss: 0.00695, Accuracy: 28.93

EPOCH: 18
Loss = 1.88 | Batch = 195 | Accuracy = 29.40: 100% 196/196 [00:14<00:00, 13.72it/s]

Test set: Average loss: 0.00690, Accuracy: 30.65

EPOCH: 19
Loss = 1.77 | Batch = 195 | Accuracy = 29.94: 100% 196/196 [00:14<00:00, 13.82it/s]

Test set: Average loss: 0.00679, Accuracy: 30.38

EPOCH: 20
Loss = 1.73 | Batch = 195 | Accuracy = 30.45: 100% 196/196 [00:14<00:00, 13.41it/s]

Test set: Average loss: 0.00708, Accuracy: 29.67

EPOCH: 21
Loss = 1.76 | Batch = 195 | Accuracy = 30.70: 100% 196/196 [00:14<00:00, 13.68it/s]

Test set: Average loss: 0.00692, Accuracy: 30.05

EPOCH: 22
Loss = 1.66 | Batch = 195 | Accuracy = 30.92: 100% 196/196 [00:14<00:00, 13.74it/s]

Test set: Average loss: 0.00670, Accuracy: 31.38

EPOCH: 23
Loss = 1.74 | Batch = 195 | Accuracy = 31.02: 100% 196/196 [00:14<00:00, 13.70it/s]

Test set: Average loss: 0.00670, Accuracy: 31.43


--------------------------------------------------------------------------------------------------------------------------------

# Final Model Performance

Train set: Average loss: 0.00661, Accuracy: 31.59 \
Test set: Average loss: 0.00670, Accuracy: 31.43


--------------------------------------------------------------------------------------------------------------------------------

# Learning Rate Schedule

![one_cycle_lr_plot](https://user-images.githubusercontent.com/21367838/221286681-79e4b8c2-73dc-4887-80df-8dc20a7374bc.png)


--------------------------------------------------------------------------------------------------------------------------------

# Mis-classified Images

![misclassified_images](https://user-images.githubusercontent.com/21367838/221286705-53a438a0-87c5-4b19-9df0-2bf18d9f7dc6.png)


--------------------------------------------------------------------------------------------------------------------------------

# Performance Plots

![performance_plot](https://user-images.githubusercontent.com/21367838/221286657-f9682534-a591-4fdc-8900-1e2d1bead062.png)
