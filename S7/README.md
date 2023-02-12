# Code Structure
1. S7.ipynb as assignment notebook
2. Plots folder has all the random images, mis-classified images, grad cam outputs and plots


--------------------------------------------------------------------------------------------------------------------------------

# Example Images

![training_images](https://user-images.githubusercontent.com/21367838/218301858-14487f34-aa2f-40e1-a6fa-a4f44849ba9e.png)


--------------------------------------------------------------------------------------------------------------------------------

# Final Model Architecture - ResNet18

![image](https://user-images.githubusercontent.com/21367838/218301719-f8d8d50e-cde3-471b-adfd-8e139236608b.png)



--------------------------------------------------------------------------------------------------------------------------------

# Model Training Strategy
1. Model wa trained for 20 epochs using cross entropy loss and sgd optimizer with weight decay.
2. Learning rate scheduler to decrease learning rate after every 5 epochs.
3. Random crop and cut out was implemented for data augmentation.


--------------------------------------------------------------------------------------------------------------------------------

# Training Logs

EPOCH: 0
Loss = 1.45 | Batch = 781 | Accuracy = 45.27: 100% 782/782 [00:49<00:00, 15.66it/s]

Test set: Average loss: 0.02227, Accuracy: 48.60

EPOCH: 1
Loss = 1.34 | Batch = 781 | Accuracy = 60.64: 100% 782/782 [00:50<00:00, 15.43it/s]

Test set: Average loss: 0.01590, Accuracy: 67.29

EPOCH: 2
Loss = 1.25 | Batch = 781 | Accuracy = 66.77: 100% 782/782 [00:50<00:00, 15.44it/s]

Test set: Average loss: 0.02069, Accuracy: 54.31

EPOCH: 3
Loss = 1.14 | Batch = 781 | Accuracy = 70.14: 100% 782/782 [00:51<00:00, 15.33it/s]

Test set: Average loss: 0.01661, Accuracy: 68.83

EPOCH: 4
Loss = 1.35 | Batch = 781 | Accuracy = 72.18: 100% 782/782 [00:50<00:00, 15.42it/s]

Test set: Average loss: 0.02007, Accuracy: 58.86

EPOCH: 5
Loss = 0.99 | Batch = 781 | Accuracy = 80.24: 100% 782/782 [00:50<00:00, 15.39it/s]

Test set: Average loss: 0.01205, Accuracy: 84.41

EPOCH: 6
Loss = 0.95 | Batch = 781 | Accuracy = 82.47: 100% 782/782 [00:50<00:00, 15.51it/s]

Test set: Average loss: 0.01138, Accuracy: 85.31

EPOCH: 7
Loss = 0.74 | Batch = 781 | Accuracy = 83.13: 100% 782/782 [00:50<00:00, 15.53it/s]

Test set: Average loss: 0.01101, Accuracy: 85.23

EPOCH: 8
Loss = 1.00 | Batch = 781 | Accuracy = 83.74: 100% 782/782 [00:50<00:00, 15.44it/s]

Test set: Average loss: 0.01127, Accuracy: 84.00

EPOCH: 9
Loss = 1.01 | Batch = 781 | Accuracy = 84.35: 100% 782/782 [00:51<00:00, 15.27it/s]

Test set: Average loss: 0.01066, Accuracy: 85.56

EPOCH: 10
Loss = 0.57 | Batch = 781 | Accuracy = 86.58: 100% 782/782 [00:50<00:00, 15.36it/s]

Test set: Average loss: 0.01003, Accuracy: 88.23

EPOCH: 11
Loss = 0.87 | Batch = 781 | Accuracy = 87.31: 100% 782/782 [00:50<00:00, 15.33it/s]

Test set: Average loss: 0.00965, Accuracy: 88.36

EPOCH: 12
Loss = 0.78 | Batch = 781 | Accuracy = 87.62: 100% 782/782 [00:50<00:00, 15.38it/s]

Test set: Average loss: 0.00976, Accuracy: 88.51

EPOCH: 13
Loss = 0.85 | Batch = 781 | Accuracy = 87.65: 100% 782/782 [00:50<00:00, 15.49it/s]

Test set: Average loss: 0.00967, Accuracy: 88.71

EPOCH: 14
Loss = 0.92 | Batch = 781 | Accuracy = 87.81: 100% 782/782 [00:50<00:00, 15.53it/s]

Test set: Average loss: 0.00966, Accuracy: 88.34

EPOCH: 15
Loss = 0.88 | Batch = 781 | Accuracy = 88.11: 100% 782/782 [00:50<00:00, 15.38it/s]

Test set: Average loss: 0.00958, Accuracy: 88.85

EPOCH: 16
Loss = 0.73 | Batch = 781 | Accuracy = 88.06: 100% 782/782 [00:51<00:00, 15.33it/s]

Test set: Average loss: 0.00963, Accuracy: 88.81

EPOCH: 17
Loss = 0.89 | Batch = 781 | Accuracy = 88.23: 100% 782/782 [00:50<00:00, 15.41it/s]

Test set: Average loss: 0.00964, Accuracy: 88.94

EPOCH: 18
Loss = 0.71 | Batch = 781 | Accuracy = 88.32: 100% 782/782 [00:50<00:00, 15.37it/s]

Test set: Average loss: 0.00960, Accuracy: 88.92

EPOCH: 19
Loss = 0.80 | Batch = 781 | Accuracy = 88.20: 100% 782/782 [00:50<00:00, 15.51it/s]

Test set: Average loss: 0.00973, Accuracy: 88.72


--------------------------------------------------------------------------------------------------------------------------------

# Final Model Performance

Train set: Average loss: 0.00971, Accuracy: 90.11 \
Test set: Average loss: 0.00973, Accuracy: 88.72


--------------------------------------------------------------------------------------------------------------------------------

# Performance Plot

![performance_plot](https://user-images.githubusercontent.com/21367838/218301900-f44dc2f5-922d-4cbd-b735-514d838167e3.png)


--------------------------------------------------------------------------------------------------------------------------------

# Mis-classified Images

![misclassified_images (1)](https://user-images.githubusercontent.com/21367838/218301871-9979050d-a77c-4c7d-ad86-22a7f466f28d.png)


--------------------------------------------------------------------------------------------------------------------------------

# GradCAM Output for Mis-classified Images Against True Label

![gradcam_true_label](https://user-images.githubusercontent.com/21367838/218301948-5b2fb393-d814-4f9c-afb2-40ee980a8014.png)


--------------------------------------------------------------------------------------------------------------------------------

# GradCAM Output for Mis-classified Images Against Prediction

![gradcam_predicted_label](https://user-images.githubusercontent.com/21367838/218301952-e45d5e91-c4af-45ad-8df4-12c2ecad1a56.png)
