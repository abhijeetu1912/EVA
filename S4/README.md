# STEPS
1. Build the basic skeleton
2. Build lighter model
3. Add batch normalization & dropout
4. Add data augmentation
5. Add Learning rate scheduler

--------------------------------------------------------------------------------------------------------------------------------

# Final Model Architecture - Uses batch norm, dropout, data augmentation & learning rate scheduler

1. Convolution layer 1: kernel size = 5 x 5, input = 1  x  28  x  28, output = 96  x  28  x  28, rf = 585 | conv -> batch norm -> relu
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

# STEP 1 - Build the basic skeleton 

Target:
Build the code skeleton and achieve a decent accuracy

Results:
Parameters: 185,930 \
Best Train Accuracy: 99.56 \
Best Test Accuracy: 99.13

Analysis:
Overfitting as performance on test data is lower. \
Training is not stable as accuracy of 99.23 was reached in 8th epoch but afterwards accuracy keeps changing from low to high.


EPOCH: 0
Loss=0.08773600310087204 Batch_id=937 Accuracy=63.66: 100%|██████████| 938/938 [00:18<00:00, 50.11it/s]

Test set: Average loss: 0.1294, Accuracy: 9587/10000 (95.87%)

EPOCH: 1
Loss=0.03443032503128052 Batch_id=937 Accuracy=97.18: 100%|██████████| 938/938 [00:19<00:00, 48.09it/s]

Test set: Average loss: 0.0672, Accuracy: 9779/10000 (97.79%)

EPOCH: 2
Loss=0.01677856594324112 Batch_id=937 Accuracy=98.12: 100%|██████████| 938/938 [00:16<00:00, 57.00it/s]

Test set: Average loss: 0.0586, Accuracy: 9819/10000 (98.19%)

EPOCH: 3
Loss=0.16868261992931366 Batch_id=937 Accuracy=98.64: 100%|██████████| 938/938 [00:16<00:00, 56.76it/s]

Test set: Average loss: 0.0356, Accuracy: 9880/10000 (98.80%)

EPOCH: 4
Loss=0.00045392828178592026 Batch_id=937 Accuracy=98.87: 100%|██████████| 938/938 [00:16<00:00, 57.34it/s]

Test set: Average loss: 0.0364, Accuracy: 9875/10000 (98.75%)

EPOCH: 5
Loss=0.10763448476791382 Batch_id=937 Accuracy=99.01: 100%|██████████| 938/938 [00:16<00:00, 56.97it/s]

Test set: Average loss: 0.0332, Accuracy: 9906/10000 (99.06%)

EPOCH: 6
Loss=0.002495982451364398 Batch_id=937 Accuracy=99.14: 100%|██████████| 938/938 [00:17<00:00, 54.49it/s]

Test set: Average loss: 0.0300, Accuracy: 9898/10000 (98.98%)

EPOCH: 7
Loss=0.0007368073565885425 Batch_id=937 Accuracy=99.33: 100%|██████████| 938/938 [00:16<00:00, 56.70it/s]

Test set: Average loss: 0.0262, Accuracy: 9923/10000 (99.23%)

EPOCH: 8
Loss=0.01737082377076149 Batch_id=937 Accuracy=99.27: 100%|██████████| 938/938 [00:16<00:00, 57.00it/s]

Test set: Average loss: 0.0315, Accuracy: 9894/10000 (98.94%)

EPOCH: 9
Loss=0.014331125654280186 Batch_id=937 Accuracy=99.39: 100%|██████████| 938/938 [00:16<00:00, 55.64it/s]

Test set: Average loss: 0.0290, Accuracy: 9904/10000 (99.04%)

EPOCH: 10
Loss=0.014854123815894127 Batch_id=937 Accuracy=99.47: 100%|██████████| 938/938 [00:16<00:00, 57.25it/s]

Test set: Average loss: 0.0288, Accuracy: 9904/10000 (99.04%)

EPOCH: 11
Loss=0.007806533947587013 Batch_id=937 Accuracy=99.46: 100%|██████████| 938/938 [00:17<00:00, 52.98it/s]

Test set: Average loss: 0.0286, Accuracy: 9904/10000 (99.04%)

EPOCH: 12
Loss=8.292205166071653e-05 Batch_id=937 Accuracy=99.61: 100%|██████████| 938/938 [00:16<00:00, 56.49it/s]

Test set: Average loss: 0.0264, Accuracy: 9916/10000 (99.16%)

EPOCH: 13
Loss=0.0002022672852035612 Batch_id=937 Accuracy=99.52: 100%|██████████| 938/938 [00:16<00:00, 56.07it/s]

Test set: Average loss: 0.0259, Accuracy: 9916/10000 (99.16%)

EPOCH: 14
Loss=0.03677373751997948 Batch_id=937 Accuracy=99.61: 100%|██████████| 938/938 [00:16<00:00, 56.26it/s]

Test set: Average loss: 0.0298, Accuracy: 9913/10000 (99.13%)


--------------------------------------------------------------------------------------------------------------------------------


# STEP 2 - Build lighter model

Target:
Build a lighter model with less than 10k parameter

Results:
Parameters: 9,122 \
Best Train Accuracy: 99.20 \
Best Test Accuracy: 98.92

Analysis:
Overfitting of lower degree as performance on test data is lower. \
Model was unable to reach the targetof 99.40 accuracy, needs some improvement in performance as well as stability.


EPOCH: 0
Loss=0.2808218002319336 Batch_id=937 Accuracy=74.59: 100%|██████████| 938/938 [00:18<00:00, 49.81it/s]

Test set: Average loss: 0.1389, Accuracy: 9567/10000 (95.67%)

EPOCH: 1
Loss=0.12248538434505463 Batch_id=937 Accuracy=96.44: 100%|██████████| 938/938 [00:17<00:00, 52.63it/s]

Test set: Average loss: 0.0961, Accuracy: 9676/10000 (96.76%)

EPOCH: 2
Loss=0.03541843593120575 Batch_id=937 Accuracy=97.33: 100%|██████████| 938/938 [00:15<00:00, 59.47it/s]

Test set: Average loss: 0.0548, Accuracy: 9821/10000 (98.21%)

EPOCH: 3
Loss=0.0024993405677378178 Batch_id=937 Accuracy=97.74: 100%|██████████| 938/938 [00:15<00:00, 59.67it/s]

Test set: Average loss: 0.0516, Accuracy: 9828/10000 (98.28%)

EPOCH: 4
Loss=0.007975039072334766 Batch_id=937 Accuracy=98.04: 100%|██████████| 938/938 [00:15<00:00, 59.06it/s]

Test set: Average loss: 0.0483, Accuracy: 9849/10000 (98.49%)

EPOCH: 5
Loss=0.009895645081996918 Batch_id=937 Accuracy=98.20: 100%|██████████| 938/938 [00:16<00:00, 57.56it/s]

Test set: Average loss: 0.0408, Accuracy: 9863/10000 (98.63%)

EPOCH: 6
Loss=0.09514982253313065 Batch_id=937 Accuracy=98.44: 100%|██████████| 938/938 [00:17<00:00, 54.76it/s]

Test set: Average loss: 0.0383, Accuracy: 9877/10000 (98.77%)

EPOCH: 7
Loss=0.0399777814745903 Batch_id=937 Accuracy=98.60: 100%|██████████| 938/938 [00:17<00:00, 52.77it/s]

Test set: Average loss: 0.0447, Accuracy: 9850/10000 (98.50%)

EPOCH: 8
Loss=0.02068287506699562 Batch_id=937 Accuracy=98.68: 100%|██████████| 938/938 [00:16<00:00, 57.33it/s]

Test set: Average loss: 0.0363, Accuracy: 9886/10000 (98.86%)

EPOCH: 9
Loss=0.017010947689414024 Batch_id=937 Accuracy=98.80: 100%|██████████| 938/938 [00:16<00:00, 56.56it/s]

Test set: Average loss: 0.0358, Accuracy: 9881/10000 (98.81%)

EPOCH: 10
Loss=0.04609169438481331 Batch_id=937 Accuracy=98.94: 100%|██████████| 938/938 [00:16<00:00, 57.02it/s]

Test set: Average loss: 0.0352, Accuracy: 9889/10000 (98.89%)

EPOCH: 11
Loss=0.0706559345126152 Batch_id=937 Accuracy=98.95: 100%|██████████| 938/938 [00:16<00:00, 57.79it/s]

Test set: Average loss: 0.0364, Accuracy: 9895/10000 (98.95%)

EPOCH: 12
Loss=0.0006923212204128504 Batch_id=937 Accuracy=98.98: 100%|██████████| 938/938 [00:16<00:00, 58.38it/s]

Test set: Average loss: 0.0293, Accuracy: 9910/10000 (99.10%)

EPOCH: 13
Loss=0.08996225893497467 Batch_id=937 Accuracy=99.10: 100%|██████████| 938/938 [00:16<00:00, 57.73it/s]

Test set: Average loss: 0.0366, Accuracy: 9880/10000 (98.80%)

EPOCH: 14
Loss=0.0003141901979688555 Batch_id=937 Accuracy=99.13: 100%|██████████| 938/938 [00:16<00:00, 55.71it/s]

Test set: Average loss: 0.0344, Accuracy: 9892/10000 (98.92%)


--------------------------------------------------------------------------------------------------------------------------------


# STEP 3 - Add batch normalization & dropout

Target:
Improve the performance of the model by adding batchnormalization. Also look for some stability in training by adding dropout.

Results:
Parameters: 9,426 \
Best Train Accuracy: 99.62 \
Best Test Accuracy: 99.37 (99.45 in 12th epoch)

Analysis:
Final test accuracy is very close to target of 99.40. \
Overfitting as performance on test data is lower. \
Training is not stable as accuracy of 99.45 was reached in 12th epoch but afterwards accuracy dropped. Hence, further stability in training is still required.


EPOCH: 0
Loss=0.06583476066589355 Batch_id=937 Accuracy=94.48: 100%|██████████| 938/938 [00:20<00:00, 46.13it/s]

Test set: Average loss: 0.0659, Accuracy: 9782/10000 (97.82%)

EPOCH: 1
Loss=0.03000473603606224 Batch_id=937 Accuracy=98.14: 100%|██████████| 938/938 [00:18<00:00, 50.65it/s]

Test set: Average loss: 0.0393, Accuracy: 9881/10000 (98.81%)

EPOCH: 2
Loss=0.005860371980816126 Batch_id=937 Accuracy=98.46: 100%|██████████| 938/938 [00:16<00:00, 55.38it/s]

Test set: Average loss: 0.0318, Accuracy: 9907/10000 (99.07%)

EPOCH: 3
Loss=0.004802419804036617 Batch_id=937 Accuracy=98.69: 100%|██████████| 938/938 [00:17<00:00, 54.99it/s]

Test set: Average loss: 0.0267, Accuracy: 9919/10000 (99.19%)

EPOCH: 4
Loss=0.00919875968247652 Batch_id=937 Accuracy=98.83: 100%|██████████| 938/938 [00:16<00:00, 56.26it/s]

Test set: Average loss: 0.0243, Accuracy: 9917/10000 (99.17%)

EPOCH: 5
Loss=0.0447070337831974 Batch_id=937 Accuracy=98.86: 100%|██████████| 938/938 [00:17<00:00, 54.96it/s]

Test set: Average loss: 0.0279, Accuracy: 9906/10000 (99.06%)

EPOCH: 6
Loss=0.03710073232650757 Batch_id=937 Accuracy=98.96: 100%|██████████| 938/938 [00:17<00:00, 52.86it/s]

Test set: Average loss: 0.0252, Accuracy: 9922/10000 (99.22%)

EPOCH: 7
Loss=0.003360309172421694 Batch_id=937 Accuracy=99.03: 100%|██████████| 938/938 [00:17<00:00, 54.42it/s]

Test set: Average loss: 0.0293, Accuracy: 9905/10000 (99.05%)

EPOCH: 8
Loss=0.011403288692235947 Batch_id=937 Accuracy=99.05: 100%|██████████| 938/938 [00:17<00:00, 54.61it/s]

Test set: Average loss: 0.0220, Accuracy: 9933/10000 (99.33%)

EPOCH: 9
Loss=0.021365936845541 Batch_id=937 Accuracy=99.17: 100%|██████████| 938/938 [00:17<00:00, 55.15it/s]

Test set: Average loss: 0.0244, Accuracy: 9926/10000 (99.26%)

EPOCH: 10
Loss=0.011438267305493355 Batch_id=937 Accuracy=99.19: 100%|██████████| 938/938 [00:16<00:00, 57.03it/s]

Test set: Average loss: 0.0212, Accuracy: 9932/10000 (99.32%)

EPOCH: 11
Loss=0.025535114109516144 Batch_id=937 Accuracy=99.24: 100%|██████████| 938/938 [00:16<00:00, 55.34it/s]

Test set: Average loss: 0.0189, Accuracy: 9945/10000 (99.45%)

EPOCH: 12
Loss=0.0008350724820047617 Batch_id=937 Accuracy=99.27: 100%|██████████| 938/938 [00:16<00:00, 55.29it/s]

Test set: Average loss: 0.0212, Accuracy: 9927/10000 (99.27%)

EPOCH: 13
Loss=0.04395304620265961 Batch_id=937 Accuracy=99.19: 100%|██████████| 938/938 [00:16<00:00, 55.46it/s]

Test set: Average loss: 0.0196, Accuracy: 9940/10000 (99.40%)

EPOCH: 14
Loss=0.0018471162766218185 Batch_id=937 Accuracy=99.28: 100%|██████████| 938/938 [00:17<00:00, 54.60it/s]

Test set: Average loss: 0.0197, Accuracy: 9937/10000 (99.37%)


--------------------------------------------------------------------------------------------------------------------------------


# STEP 4 - Add data augmentation

Target:
Improve the performance on test data and improve training stability by adding data augmentation.

Results:
Parameters: 9,426 \
Best Train Accuracy: 99.40 \
Best Test Accuracy: 99.39 (99.41 in 8th epoch)

Analysis:
No Overfitting on test data, performance on test data has improved by a margin. \
Training is not stable as accuracy of 99.41 was reached in 8th epoch but afterwards accuracy dropped.


EPOCH: 0
Loss=0.05966421589255333 Batch_id=937 Accuracy=94.11: 100%|██████████| 938/938 [00:24<00:00, 37.71it/s]

Test set: Average loss: 0.0490, Accuracy: 9855/10000 (98.55%)

EPOCH: 1
Loss=0.06825853139162064 Batch_id=937 Accuracy=97.88: 100%|██████████| 938/938 [00:20<00:00, 45.72it/s]

Test set: Average loss: 0.0425, Accuracy: 9871/10000 (98.71%)

EPOCH: 2
Loss=0.009801664389669895 Batch_id=937 Accuracy=98.22: 100%|██████████| 938/938 [00:20<00:00, 45.72it/s]

Test set: Average loss: 0.0273, Accuracy: 9912/10000 (99.12%)

EPOCH: 3
Loss=0.0032393685542047024 Batch_id=937 Accuracy=98.34: 100%|██████████| 938/938 [00:20<00:00, 46.06it/s]

Test set: Average loss: 0.0289, Accuracy: 9901/10000 (99.01%)

EPOCH: 4
Loss=0.009570731781423092 Batch_id=937 Accuracy=98.54: 100%|██████████| 938/938 [00:21<00:00, 43.80it/s]

Test set: Average loss: 0.0250, Accuracy: 9921/10000 (99.21%)

EPOCH: 5
Loss=0.06762854009866714 Batch_id=937 Accuracy=98.60: 100%|██████████| 938/938 [00:20<00:00, 44.99it/s]

Test set: Average loss: 0.0269, Accuracy: 9913/10000 (99.13%)

EPOCH: 6
Loss=0.05752908065915108 Batch_id=937 Accuracy=98.69: 100%|██████████| 938/938 [00:21<00:00, 44.19it/s]

Test set: Average loss: 0.0225, Accuracy: 9930/10000 (99.30%)

EPOCH: 7
Loss=0.0074791088700294495 Batch_id=937 Accuracy=98.86: 100%|██████████| 938/938 [00:20<00:00, 45.11it/s]

Test set: Average loss: 0.0216, Accuracy: 9941/10000 (99.41%)

EPOCH: 8
Loss=0.050894275307655334 Batch_id=937 Accuracy=98.84: 100%|██████████| 938/938 [00:20<00:00, 45.48it/s]

Test set: Average loss: 0.0206, Accuracy: 9937/10000 (99.37%)

EPOCH: 9
Loss=0.020080821588635445 Batch_id=937 Accuracy=98.88: 100%|██████████| 938/938 [00:20<00:00, 45.31it/s]

Test set: Average loss: 0.0212, Accuracy: 9938/10000 (99.38%)

EPOCH: 10
Loss=0.00800067838281393 Batch_id=937 Accuracy=98.91: 100%|██████████| 938/938 [00:20<00:00, 45.48it/s]

Test set: Average loss: 0.0236, Accuracy: 9926/10000 (99.26%)

EPOCH: 11
Loss=0.024968568235635757 Batch_id=937 Accuracy=98.98: 100%|██████████| 938/938 [00:20<00:00, 45.15it/s]

Test set: Average loss: 0.0214, Accuracy: 9929/10000 (99.29%)

EPOCH: 12
Loss=0.003664381103590131 Batch_id=937 Accuracy=98.94: 100%|██████████| 938/938 [00:20<00:00, 45.60it/s]

Test set: Average loss: 0.0209, Accuracy: 9931/10000 (99.31%)

EPOCH: 13
Loss=0.03816790506243706 Batch_id=937 Accuracy=99.03: 100%|██████████| 938/938 [00:20<00:00, 45.54it/s]

Test set: Average loss: 0.0192, Accuracy: 9939/10000 (99.39%)

EPOCH: 14
Loss=0.0032763141207396984 Batch_id=937 Accuracy=99.10: 100%|██████████| 938/938 [00:21<00:00, 43.03it/s]

Test set: Average loss: 0.0194, Accuracy: 9939/10000 (99.39%)


--------------------------------------------------------------------------------------------------------------------------------


# STEP 5 - Add Learning rate scheduler

Target:
Stabilize the model training and achieve the desired performance of 99.40 on test data.

Results:
Parameters: 9,426
Best Train Accuracy: 99.33
Best Test Accuracy: 99.40

Analysis:
No Overfitting as performance on test data is close to train data.
Training is stable as accuracy keeps increasing with every epoch and remains stable. Achived 99.40 test accuracy for last 3 epochs and 99.30-99.38 in earlier epochs.


EPOCH: 0
Loss=0.04666141793131828 Batch_id=937 Accuracy=94.18: 100%|██████████| 938/938 [00:08<00:00, 106.57it/s]

Test set: Average loss: 0.0565, Accuracy: 9818/10000 (98.18%)

EPOCH: 1
Loss=0.09398200362920761 Batch_id=937 Accuracy=97.84: 100%|██████████| 938/938 [00:07<00:00, 118.29it/s]

Test set: Average loss: 0.0393, Accuracy: 9881/10000 (98.81%)

EPOCH: 2
Loss=0.014432268217206001 Batch_id=937 Accuracy=98.16: 100%|██████████| 938/938 [00:07<00:00, 121.17it/s]

Test set: Average loss: 0.0306, Accuracy: 9909/10000 (99.09%)

EPOCH: 3
Loss=0.006781352683901787 Batch_id=937 Accuracy=98.36: 100%|██████████| 938/938 [00:07<00:00, 123.05it/s]

Test set: Average loss: 0.0292, Accuracy: 9917/10000 (99.17%)

EPOCH: 4
Loss=0.007954735308885574 Batch_id=937 Accuracy=98.55: 100%|██████████| 938/938 [00:07<00:00, 121.83it/s]

Test set: Average loss: 0.0319, Accuracy: 9901/10000 (99.01%)

EPOCH: 5
Loss=0.04459052160382271 Batch_id=937 Accuracy=98.79: 100%|██████████| 938/938 [00:07<00:00, 121.09it/s]

Test set: Average loss: 0.0216, Accuracy: 9936/10000 (99.36%)

EPOCH: 6
Loss=0.02837185189127922 Batch_id=937 Accuracy=98.95: 100%|██████████| 938/938 [00:07<00:00, 120.90it/s]

Test set: Average loss: 0.0211, Accuracy: 9938/10000 (99.38%)

EPOCH: 7
Loss=0.02394060790538788 Batch_id=937 Accuracy=98.98: 100%|██████████| 938/938 [00:07<00:00, 120.56it/s]

Test set: Average loss: 0.0206, Accuracy: 9934/10000 (99.34%)

EPOCH: 8
Loss=0.02290717139840126 Batch_id=937 Accuracy=99.02: 100%|██████████| 938/938 [00:07<00:00, 119.36it/s]

Test set: Average loss: 0.0201, Accuracy: 9936/10000 (99.36%)

EPOCH: 9
Loss=0.035111334174871445 Batch_id=937 Accuracy=99.02: 100%|██████████| 938/938 [00:07<00:00, 121.70it/s]

Test set: Average loss: 0.0204, Accuracy: 9934/10000 (99.34%)

EPOCH: 10
Loss=0.0072777424938976765 Batch_id=937 Accuracy=99.05: 100%|██████████| 938/938 [00:07<00:00, 122.14it/s]

Test set: Average loss: 0.0202, Accuracy: 9935/10000 (99.35%)

EPOCH: 11
Loss=0.052780941128730774 Batch_id=937 Accuracy=99.04: 100%|██████████| 938/938 [00:07<00:00, 120.60it/s]

Test set: Average loss: 0.0199, Accuracy: 9936/10000 (99.36%)

EPOCH: 12
Loss=0.0023615683894604445 Batch_id=937 Accuracy=99.04: 100%|██████████| 938/938 [00:07<00:00, 120.86it/s]

Test set: Average loss: 0.0196, Accuracy: 9940/10000 (99.40%)

EPOCH: 13
Loss=0.04156742990016937 Batch_id=937 Accuracy=99.03: 100%|██████████| 938/938 [00:07<00:00, 120.17it/s]

Test set: Average loss: 0.0197, Accuracy: 9940/10000 (99.40%)

EPOCH: 14
Loss=0.004937621299177408 Batch_id=937 Accuracy=99.00: 100%|██████████| 938/938 [00:07<00:00, 117.78it/s]

Test set: Average loss: 0.0200, Accuracy: 9940/10000 (99.40%)
