# Methods
1. Batch Normalization + L1 Regularization (lambda = 0.001): 
   We calculate mean and std dev of each channel in batch normalization.
3. Layer Normalization: 
   We calculate the mean and the variance of each image in the mini-batch for all channels.
5. Group Normalization with Groups = Channels / 4: 
   We calculate the mean and the variance for each image in each group.
   
--------------------------------------------------------------------------------------------------------------------------------

# Final Model Architecture - Uses normalizationm, dropout & learning rate scheduler

1. Convolution layer 1: kernel size = 5 x 5, input = 1  x  28  x  28, output = 96  x  28  x  28, rf = 5 x 5 | conv -> batch norm -> relu
2. Convolution layer 2: kernel size = 1 x 1, input = 96  x  28  x  28, output = 16  x  28  x  28, rf = 5 x 5 | conv -> batch norm -> relu
3. Max pooling of kernel size 2 and stride 2: input = 16  x  28  x  28, output = 16  x  14  x  14, rf = 6 x 6 | max pool -> dropout
4. Convolution layer 3: kernel size = 3 x 3, input = 16  x  14  x  14, output = 16  x  12  x  12, rf = 10 x 10 | conv -> batch norm -> relu
5. Convolution layer 4: kernel size = 1 x 1, input = 16  x  12  x  12, output = 8  x  12  x  12, rf = 10 x 10 | conv -> batch norm -> relu
6. Max pooling of kernel size 2 and stride 2: input = 8  x  12  x  12, output = 8  x  6  x  6, rf = 12 x 12 | max pool -> dropout
7. Convolution layer 5: kernel size = 3 x 3, input = 8  x  6  x  6, output = 16  x  4  x  4, rf = 20 x 20 | conv -> batch norm -> relu
8. Convolution layer 6: kernel size = 3 x 3, input = 16  x  4  x  4, output = 10  x  2  x  2, rf = 28 x 28 | Relu is not applied after last conv layer
9. Global average pooling (gap): input = 10  x  2  x 2, output = 10  x  1  x  1, rf = 32 x 32
10. Softmax on gap outputs

<img width="361" alt="image" src="https://user-images.githubusercontent.com/21367838/213840811-c5979b95-ec95-401c-ba20-5a8704e26555.png">


--------------------------------------------------------------------------------------------------------------------------------

# STEP 1 - Batch Normalization + L1

## Logs

EPOCH: 0  LR:  0.01
Loss = 0.58 | Batch = 937 | Accuracy = 93.83: 100%|██████████| 938/938 [00:32<00:00, 28.69it/s]

Test set: Average loss: 0.00126, Accuracy: 97.66

EPOCH: 1  LR:  0.01
Loss = 0.42 | Batch = 937 | Accuracy = 96.96: 100%|██████████| 938/938 [00:30<00:00, 31.15it/s]

Test set: Average loss: 0.00131, Accuracy: 97.79

EPOCH: 2  LR:  0.01
Loss = 0.38 | Batch = 937 | Accuracy = 96.91: 100%|██████████| 938/938 [00:29<00:00, 31.76it/s]

Test set: Average loss: 0.00147, Accuracy: 97.40

EPOCH: 3  LR:  0.01
Loss = 0.37 | Batch = 937 | Accuracy = 96.97: 100%|██████████| 938/938 [00:29<00:00, 31.61it/s]

Test set: Average loss: 0.00098, Accuracy: 98.32

EPOCH: 4  LR:  0.01
Loss = 0.31 | Batch = 937 | Accuracy = 97.16: 100%|██████████| 938/938 [00:28<00:00, 32.64it/s]

Test set: Average loss: 0.00156, Accuracy: 96.96

EPOCH: 5  LR:  0.001
Loss = 0.30 | Batch = 937 | Accuracy = 98.07: 100%|██████████| 938/938 [00:29<00:00, 31.31it/s]

Test set: Average loss: 0.00060, Accuracy: 98.87

EPOCH: 6  LR:  0.001
Loss = 0.28 | Batch = 937 | Accuracy = 98.31: 100%|██████████| 938/938 [00:29<00:00, 32.08it/s]

Test set: Average loss: 0.00058, Accuracy: 98.95

EPOCH: 7  LR:  0.001
Loss = 0.23 | Batch = 937 | Accuracy = 98.31: 100%|██████████| 938/938 [00:30<00:00, 30.47it/s]

Test set: Average loss: 0.00059, Accuracy: 98.88

EPOCH: 8  LR:  0.001
Loss = 0.31 | Batch = 937 | Accuracy = 98.23: 100%|██████████| 938/938 [00:30<00:00, 31.23it/s]

Test set: Average loss: 0.00059, Accuracy: 98.86

EPOCH: 9  LR:  0.001
Loss = 0.26 | Batch = 937 | Accuracy = 98.18: 100%|██████████| 938/938 [00:31<00:00, 30.07it/s]

Test set: Average loss: 0.00066, Accuracy: 98.77

EPOCH: 10  LR:  0.0001
Loss = 0.26 | Batch = 937 | Accuracy = 98.44: 100%|██████████| 938/938 [00:29<00:00, 31.50it/s]

Test set: Average loss: 0.00051, Accuracy: 99.18

EPOCH: 11  LR:  0.0001
Loss = 0.33 | Batch = 937 | Accuracy = 98.54: 100%|██████████| 938/938 [00:30<00:00, 31.03it/s]

Test set: Average loss: 0.00050, Accuracy: 99.19

EPOCH: 12  LR:  0.0001
Loss = 0.21 | Batch = 937 | Accuracy = 98.61: 100%|██████████| 938/938 [00:29<00:00, 31.53it/s]

Test set: Average loss: 0.00049, Accuracy: 99.19

EPOCH: 13  LR:  0.0001
Loss = 0.24 | Batch = 937 | Accuracy = 98.62: 100%|██████████| 938/938 [00:29<00:00, 31.51it/s]

Test set: Average loss: 0.00049, Accuracy: 99.22

EPOCH: 14  LR:  0.0001
Loss = 0.21 | Batch = 937 | Accuracy = 98.59: 100%|██████████| 938/938 [00:29<00:00, 31.63it/s]

Test set: Average loss: 0.00050, Accuracy: 99.22

EPOCH: 15  LR:  1e-05
Loss = 0.23 | Batch = 937 | Accuracy = 98.68: 100%|██████████| 938/938 [00:30<00:00, 31.01it/s]

Test set: Average loss: 0.00048, Accuracy: 99.20

EPOCH: 16  LR:  1e-05
Loss = 0.21 | Batch = 937 | Accuracy = 98.61: 100%|██████████| 938/938 [00:29<00:00, 31.39it/s]

Test set: Average loss: 0.00048, Accuracy: 99.21

EPOCH: 17  LR:  1e-05
Loss = 0.25 | Batch = 937 | Accuracy = 98.61: 100%|██████████| 938/938 [00:29<00:00, 31.36it/s]

Test set: Average loss: 0.00049, Accuracy: 99.15

EPOCH: 18  LR:  1e-05
Loss = 0.20 | Batch = 937 | Accuracy = 98.60: 100%|██████████| 938/938 [00:30<00:00, 30.61it/s]

Test set: Average loss: 0.00049, Accuracy: 99.21

EPOCH: 19  LR:  1e-05
Loss = 0.22 | Batch = 937 | Accuracy = 98.65: 100%|██████████| 938/938 [00:30<00:00, 31.18it/s]

Test set: Average loss: 0.00049, Accuracy: 99.23


## Final Model Performance

Train set: Average loss: 0.00058, Accuracy: 99.00 \
Test set: Average loss: 0.00048, Accuracy: 99.23

## Misclassified Images

![image](https://user-images.githubusercontent.com/21367838/221742233-cb041272-c731-4539-aa4a-972058d3c069.png)


--------------------------------------------------------------------------------------------------------------------------------

# STEP 2 - Layer Normalization

## Logs

EPOCH: 0  LR:  0.01
Loss = 0.01 | Batch = 937 | Accuracy = 91.27: 100%|██████████| 938/938 [00:28<00:00, 33.18it/s]

Test set: Average loss: 0.00102, Accuracy: 98.04

EPOCH: 1  LR:  0.01
Loss = 0.14 | Batch = 937 | Accuracy = 97.16: 100%|██████████| 938/938 [00:28<00:00, 33.25it/s]

Test set: Average loss: 0.00087, Accuracy: 98.22

EPOCH: 2  LR:  0.01
Loss = 0.18 | Batch = 937 | Accuracy = 97.80: 100%|██████████| 938/938 [00:29<00:00, 32.05it/s]

Test set: Average loss: 0.00063, Accuracy: 98.77

EPOCH: 3  LR:  0.01
Loss = 0.13 | Batch = 937 | Accuracy = 98.03: 100%|██████████| 938/938 [00:28<00:00, 32.84it/s]

Test set: Average loss: 0.00055, Accuracy: 98.81

EPOCH: 4  LR:  0.01
Loss = 0.01 | Batch = 937 | Accuracy = 98.20: 100%|██████████| 938/938 [00:28<00:00, 33.28it/s]

Test set: Average loss: 0.00049, Accuracy: 98.99

EPOCH: 5  LR:  0.001
Loss = 0.08 | Batch = 937 | Accuracy = 98.71: 100%|██████████| 938/938 [00:28<00:00, 33.49it/s]

Test set: Average loss: 0.00042, Accuracy: 99.22

EPOCH: 6  LR:  0.001
Loss = 0.01 | Batch = 937 | Accuracy = 98.71: 100%|██████████| 938/938 [00:28<00:00, 32.65it/s]

Test set: Average loss: 0.00042, Accuracy: 99.18

EPOCH: 7  LR:  0.001
Loss = 0.05 | Batch = 937 | Accuracy = 98.80: 100%|██████████| 938/938 [00:28<00:00, 33.25it/s]

Test set: Average loss: 0.00041, Accuracy: 99.24

EPOCH: 8  LR:  0.001
Loss = 0.08 | Batch = 937 | Accuracy = 98.84: 100%|██████████| 938/938 [00:28<00:00, 33.18it/s]

Test set: Average loss: 0.00040, Accuracy: 99.24

EPOCH: 9  LR:  0.001
Loss = 0.08 | Batch = 937 | Accuracy = 98.83: 100%|██████████| 938/938 [00:29<00:00, 31.28it/s]

Test set: Average loss: 0.00040, Accuracy: 99.25

EPOCH: 10  LR:  0.0001
Loss = 0.04 | Batch = 937 | Accuracy = 98.87: 100%|██████████| 938/938 [00:28<00:00, 33.23it/s]

Test set: Average loss: 0.00039, Accuracy: 99.28

EPOCH: 11  LR:  0.0001
Loss = 0.00 | Batch = 937 | Accuracy = 98.91: 100%|██████████| 938/938 [00:27<00:00, 33.77it/s]

Test set: Average loss: 0.00039, Accuracy: 99.28

EPOCH: 12  LR:  0.0001
Loss = 0.06 | Batch = 937 | Accuracy = 98.87: 100%|██████████| 938/938 [00:28<00:00, 33.15it/s]

Test set: Average loss: 0.00039, Accuracy: 99.29

EPOCH: 13  LR:  0.0001
Loss = 0.00 | Batch = 937 | Accuracy = 98.90: 100%|██████████| 938/938 [00:27<00:00, 33.57it/s]

Test set: Average loss: 0.00039, Accuracy: 99.32

EPOCH: 14  LR:  0.0001
Loss = 0.03 | Batch = 937 | Accuracy = 98.81: 100%|██████████| 938/938 [00:28<00:00, 32.75it/s]

Test set: Average loss: 0.00039, Accuracy: 99.33

EPOCH: 15  LR:  1e-05
Loss = 0.02 | Batch = 937 | Accuracy = 98.88: 100%|██████████| 938/938 [00:28<00:00, 32.67it/s]

Test set: Average loss: 0.00039, Accuracy: 99.33

EPOCH: 16  LR:  1e-05
Loss = 0.02 | Batch = 937 | Accuracy = 98.88: 100%|██████████| 938/938 [00:28<00:00, 33.44it/s]

Test set: Average loss: 0.00039, Accuracy: 99.33

EPOCH: 17  LR:  1e-05
Loss = 0.08 | Batch = 937 | Accuracy = 98.94: 100%|██████████| 938/938 [00:28<00:00, 32.73it/s]

Test set: Average loss: 0.00039, Accuracy: 99.33

EPOCH: 18  LR:  1e-05
Loss = 0.11 | Batch = 937 | Accuracy = 98.89: 100%|██████████| 938/938 [00:28<00:00, 33.28it/s]

Test set: Average loss: 0.00039, Accuracy: 99.34

EPOCH: 19  LR:  1e-05
Loss = 0.14 | Batch = 937 | Accuracy = 98.92: 100%|██████████| 938/938 [00:27<00:00, 34.50it/s]

Test set: Average loss: 0.00040, Accuracy: 99.34


## Final Model Performance

Train set: Average loss: 0.00058, Accuracy: 99.08 \
Test set: Average loss: 0.00048, Accuracy: 99.34

## Misclassified Images

![image](https://user-images.githubusercontent.com/21367838/221742843-9a46bf86-4047-434a-b3fa-76a37352383a.png)


--------------------------------------------------------------------------------------------------------------------------------

# STEP 1 - Group Normalization

## Logs

EPOCH: 0  LR:  0.01
Loss = 0.06 | Batch = 937 | Accuracy = 92.08: 100%|██████████| 938/938 [00:28<00:00, 33.27it/s]

Test set: Average loss: 0.00103, Accuracy: 97.97

EPOCH: 1  LR:  0.01
Loss = 0.04 | Batch = 937 | Accuracy = 97.21: 100%|██████████| 938/938 [00:28<00:00, 32.98it/s]

Test set: Average loss: 0.00109, Accuracy: 97.67

EPOCH: 2  LR:  0.01
Loss = 0.22 | Batch = 937 | Accuracy = 97.67: 100%|██████████| 938/938 [00:27<00:00, 33.54it/s]

Test set: Average loss: 0.00071, Accuracy: 98.64

EPOCH: 3  LR:  0.01
Loss = 0.05 | Batch = 937 | Accuracy = 97.92: 100%|██████████| 938/938 [00:29<00:00, 32.05it/s]

Test set: Average loss: 0.00064, Accuracy: 98.73

EPOCH: 4  LR:  0.01
Loss = 0.04 | Batch = 937 | Accuracy = 98.19: 100%|██████████| 938/938 [00:28<00:00, 33.27it/s]

Test set: Average loss: 0.00051, Accuracy: 98.90

EPOCH: 5  LR:  0.001
Loss = 0.02 | Batch = 937 | Accuracy = 98.61: 100%|██████████| 938/938 [00:27<00:00, 33.50it/s]

Test set: Average loss: 0.00043, Accuracy: 99.10

EPOCH: 6  LR:  0.001
Loss = 0.10 | Batch = 937 | Accuracy = 98.69: 100%|██████████| 938/938 [00:28<00:00, 33.36it/s]

Test set: Average loss: 0.00041, Accuracy: 99.13

EPOCH: 7  LR:  0.001
Loss = 0.00 | Batch = 937 | Accuracy = 98.69: 100%|██████████| 938/938 [00:28<00:00, 33.35it/s]

Test set: Average loss: 0.00041, Accuracy: 99.12

EPOCH: 8  LR:  0.001
Loss = 0.01 | Batch = 937 | Accuracy = 98.72: 100%|██████████| 938/938 [00:27<00:00, 34.07it/s]

Test set: Average loss: 0.00041, Accuracy: 99.16

EPOCH: 9  LR:  0.001
Loss = 0.16 | Batch = 937 | Accuracy = 98.84: 100%|██████████| 938/938 [00:27<00:00, 34.33it/s]

Test set: Average loss: 0.00040, Accuracy: 99.15

EPOCH: 10  LR:  0.0001
Loss = 0.00 | Batch = 937 | Accuracy = 98.81: 100%|██████████| 938/938 [00:28<00:00, 33.18it/s]

Test set: Average loss: 0.00039, Accuracy: 99.17

EPOCH: 11  LR:  0.0001
Loss = 0.00 | Batch = 937 | Accuracy = 98.81: 100%|██████████| 938/938 [00:27<00:00, 34.08it/s]

Test set: Average loss: 0.00039, Accuracy: 99.16

EPOCH: 12  LR:  0.0001
Loss = 0.00 | Batch = 937 | Accuracy = 98.83: 100%|██████████| 938/938 [00:27<00:00, 34.11it/s]

Test set: Average loss: 0.00039, Accuracy: 99.16

EPOCH: 13  LR:  0.0001
Loss = 0.00 | Batch = 937 | Accuracy = 98.87: 100%|██████████| 938/938 [00:27<00:00, 34.01it/s]

Test set: Average loss: 0.00039, Accuracy: 99.16

EPOCH: 14  LR:  0.0001
Loss = 0.05 | Batch = 937 | Accuracy = 98.85: 100%|██████████| 938/938 [00:28<00:00, 33.05it/s]

Test set: Average loss: 0.00039, Accuracy: 99.19

EPOCH: 15  LR:  1e-05
Loss = 0.03 | Batch = 937 | Accuracy = 98.81: 100%|██████████| 938/938 [00:27<00:00, 34.14it/s]

Test set: Average loss: 0.00039, Accuracy: 99.18

EPOCH: 16  LR:  1e-05
Loss = 0.01 | Batch = 937 | Accuracy = 98.89: 100%|██████████| 938/938 [00:27<00:00, 33.92it/s]

Test set: Average loss: 0.00039, Accuracy: 99.20

EPOCH: 17  LR:  1e-05
Loss = 0.01 | Batch = 937 | Accuracy = 98.84: 100%|██████████| 938/938 [00:27<00:00, 33.50it/s]

Test set: Average loss: 0.00039, Accuracy: 99.19

EPOCH: 18  LR:  1e-05
Loss = 0.00 | Batch = 937 | Accuracy = 98.77: 100%|██████████| 938/938 [00:27<00:00, 33.63it/s]

Test set: Average loss: 0.00039, Accuracy: 99.18

EPOCH: 19  LR:  1e-05
Loss = 0.00 | Batch = 937 | Accuracy = 98.87: 100%|██████████| 938/938 [00:27<00:00, 34.06it/s]

Test set: Average loss: 0.00039, Accuracy: 99.17


## Final Model Performance

Train set: Average loss: 0.00058, Accuracy: 99.07 \
Test set: Average loss: 0.00048, Accuracy: 99.17

## Misclassified Images

![image](https://user-images.githubusercontent.com/21367838/221743040-a1233f04-cc22-45e9-9a7c-6b704eaf4c5b.png)


--------------------------------------------------------------------------------------------------------------------------------

# Performance Plots

## Validation Loss

![image](https://user-images.githubusercontent.com/21367838/221743166-87e806d7-acf7-47fa-9649-ec233c9c824c.png)

## Validation Accuracy

![image](https://user-images.githubusercontent.com/21367838/221743229-3770e7c5-7151-4921-a023-304740c26073.png)


--------------------------------------------------------------------------------------------------------------------------------

# Normaztion Calculations

Refer to the file - Normalization_Calculations.xlsx

![image](https://user-images.githubusercontent.com/21367838/221749049-1298175a-1302-48e3-b664-50233a06a42b.png)

![image](https://user-images.githubusercontent.com/21367838/221749085-a5d5c263-48a5-48ed-ae8b-7edbac95868b.png)


--------------------------------------------------------------------------------------------------------------------------------

# Findings

In current setup Layer normalization had the highest performance. But still its performance was lesser than plain batchnormalization. Also it resulted in better regularization. Group normalization didn't bring any improvements.
