# Code Structure
1. File model/model.py has model structure
2. File utils/utils.py has code to show image, train, test and evaluate model
3. File main.py has everything from data loader to model training, evaluation by refererring code from model/model.py & utils/utils.py.
4. Folder plots has photoes with random image examples as well as mis classified examples.


--------------------------------------------------------------------------------------------------------------------------------

# Example Images

![random_examples](https://user-images.githubusercontent.com/21367838/217028212-5021b5a9-b668-4e84-854d-61adb732a488.png)


--------------------------------------------------------------------------------------------------------------------------------

# Final Model Architecture - Uses traditional convolution

1. Convolution block 1: 3 convolution layers of kernel size 3 x 3 and padding = 1. 3rd convolution layer here has stride 2 to act like max pooling. Dropout is applied after 3rd convolution layer.
2. Convolution block 2: Depthwise separable convolution layer with 3 x 3 and 1 x 1 kernel, follwed by a dilated convolution layer followed by convolution with stride = 2. Dropout after each convolution layer.
3. Convolution block 2: Depthwise separable convolution layer with 3 x 3 and 1 x 1 kernel, follwed by a dilated convolution layer followed by convolution with stride = 2. Dropout after each convolution layer.
4. Convolution block 4: Single convolution layer with kernel 3 x 3.
5. Global average pooling layer to reduce the size of channel. Afterwards gap output is squeezed and log softmax is applied on it.


![image](https://user-images.githubusercontent.com/21367838/217025982-b6685d88-a1b5-47f7-b4d4-f9987e47491d.png)


--------------------------------------------------------------------------------------------------------------------------------

# Model Training Strategy
1. Model wa strained for 60 epochs using cross entropy loss and rmsprop optimizer with default parameters.
2. Learning rate scheduler to decrease learning rate if test accuracy doesn't improve for 5 epochs.
3. Data augmentation stragey was applied and horizontal flip, shift, scale, rotate and coarse dropout method was used.


--------------------------------------------------------------------------------------------------------------------------------

# Training Logs

EPOCH: 0
Loss=1.0371882915496826 Batch_id=3124 Accuracy=45.44: 100% 3125/3125 [00:47<00:00, 66.03it/s]

Test set: Average loss: 0.06605, Accuracy: 6212/10000 (62.12%)

EPOCH: 1
Loss=1.3567057847976685 Batch_id=3124 Accuracy=59.46: 100% 3125/3125 [00:47<00:00, 66.46it/s]

Test set: Average loss: 0.05686, Accuracy: 6820/10000 (68.20%)

EPOCH: 2
Loss=0.6340452432632446 Batch_id=3124 Accuracy=64.77: 100% 3125/3125 [00:46<00:00, 67.69it/s]

Test set: Average loss: 0.05901, Accuracy: 6741/10000 (67.41%)

EPOCH: 3
Loss=0.9716295003890991 Batch_id=3124 Accuracy=67.58: 100% 3125/3125 [00:48<00:00, 64.51it/s]

Test set: Average loss: 0.04948, Accuracy: 7270/10000 (72.70%)

EPOCH: 4
Loss=0.84260493516922 Batch_id=3124 Accuracy=69.87: 100% 3125/3125 [00:46<00:00, 67.14it/s]

Test set: Average loss: 0.04841, Accuracy: 7272/10000 (72.72%)

EPOCH: 5
Loss=0.6307776570320129 Batch_id=3124 Accuracy=71.22: 100% 3125/3125 [00:46<00:00, 67.69it/s]

Test set: Average loss: 0.04279, Accuracy: 7592/10000 (75.92%)

EPOCH: 6
Loss=1.4117980003356934 Batch_id=3124 Accuracy=72.30: 100% 3125/3125 [00:47<00:00, 65.45it/s]

Test set: Average loss: 0.04143, Accuracy: 7708/10000 (77.08%)

EPOCH: 7
Loss=0.49843889474868774 Batch_id=3124 Accuracy=73.29: 100% 3125/3125 [00:46<00:00, 67.01it/s]

Test set: Average loss: 0.03912, Accuracy: 7831/10000 (78.31%)

EPOCH: 8
Loss=0.6615104675292969 Batch_id=3124 Accuracy=74.15: 100% 3125/3125 [00:47<00:00, 66.05it/s]

Test set: Average loss: 0.03899, Accuracy: 7861/10000 (78.61%)

EPOCH: 9
Loss=1.0459390878677368 Batch_id=3124 Accuracy=74.99: 100% 3125/3125 [00:46<00:00, 66.59it/s]

Test set: Average loss: 0.03709, Accuracy: 7985/10000 (79.85%)

EPOCH: 10
Loss=0.5332900285720825 Batch_id=3124 Accuracy=75.57: 100% 3125/3125 [00:47<00:00, 66.27it/s]

Test set: Average loss: 0.03560, Accuracy: 8037/10000 (80.37%)

EPOCH: 11
Loss=0.5989558100700378 Batch_id=3124 Accuracy=75.99: 100% 3125/3125 [00:46<00:00, 66.74it/s]

Test set: Average loss: 0.03427, Accuracy: 8135/10000 (81.35%)

EPOCH: 12
Loss=0.5627788305282593 Batch_id=3124 Accuracy=76.51: 100% 3125/3125 [00:45<00:00, 68.81it/s]

Test set: Average loss: 0.03449, Accuracy: 8069/10000 (80.69%)

EPOCH: 13
Loss=0.5074597597122192 Batch_id=3124 Accuracy=77.13: 100% 3125/3125 [00:48<00:00, 64.96it/s]

Test set: Average loss: 0.03479, Accuracy: 8044/10000 (80.44%)

EPOCH: 14
Loss=0.636087954044342 Batch_id=3124 Accuracy=77.62: 100% 3125/3125 [00:46<00:00, 67.71it/s]

Test set: Average loss: 0.03347, Accuracy: 8167/10000 (81.67%)

EPOCH: 15
Loss=0.6664760112762451 Batch_id=3124 Accuracy=77.99: 100% 3125/3125 [00:46<00:00, 67.05it/s]

Test set: Average loss: 0.03330, Accuracy: 8179/10000 (81.79%)

EPOCH: 16
Loss=0.9248341917991638 Batch_id=3124 Accuracy=78.24: 100% 3125/3125 [00:47<00:00, 66.11it/s]

Test set: Average loss: 0.03178, Accuracy: 8269/10000 (82.69%)

EPOCH: 17
Loss=0.5888480544090271 Batch_id=3124 Accuracy=78.46: 100% 3125/3125 [00:46<00:00, 67.28it/s]

Test set: Average loss: 0.03267, Accuracy: 8247/10000 (82.47%)

EPOCH: 18
Loss=0.5826719999313354 Batch_id=3124 Accuracy=79.07: 100% 3125/3125 [00:46<00:00, 66.57it/s]

Test set: Average loss: 0.03321, Accuracy: 8197/10000 (81.97%)

EPOCH: 19
Loss=0.7042016386985779 Batch_id=3124 Accuracy=78.94: 100% 3125/3125 [00:47<00:00, 65.69it/s]

Test set: Average loss: 0.03202, Accuracy: 8262/10000 (82.62%)

EPOCH: 20
Loss=0.7681995034217834 Batch_id=3124 Accuracy=79.25: 100% 3125/3125 [00:46<00:00, 66.72it/s]

Test set: Average loss: 0.03359, Accuracy: 8181/10000 (81.81%)

EPOCH: 21
Loss=0.533341109752655 Batch_id=3124 Accuracy=81.21: 100% 3125/3125 [00:45<00:00, 67.94it/s]

Test set: Average loss: 0.02847, Accuracy: 8459/10000 (84.59%)

EPOCH: 22
Loss=0.3705846667289734 Batch_id=3124 Accuracy=81.73: 100% 3125/3125 [00:47<00:00, 66.00it/s]

Test set: Average loss: 0.02811, Accuracy: 8469/10000 (84.69%)

EPOCH: 23
Loss=0.1283046454191208 Batch_id=3124 Accuracy=82.00: 100% 3125/3125 [00:47<00:00, 66.41it/s]

Test set: Average loss: 0.02847, Accuracy: 8450/10000 (84.50%)

EPOCH: 24
Loss=0.3853057622909546 Batch_id=3124 Accuracy=82.25: 100% 3125/3125 [00:46<00:00, 67.69it/s]

Test set: Average loss: 0.02819, Accuracy: 8485/10000 (84.85%)

EPOCH: 25
Loss=0.1919240802526474 Batch_id=3124 Accuracy=81.87: 100% 3125/3125 [00:48<00:00, 64.57it/s]

Test set: Average loss: 0.02796, Accuracy: 8468/10000 (84.68%)

EPOCH: 26
Loss=0.293356329202652 Batch_id=3124 Accuracy=82.40: 100% 3125/3125 [00:46<00:00, 67.68it/s]

Test set: Average loss: 0.02796, Accuracy: 8470/10000 (84.70%)

EPOCH: 27
Loss=0.666530191898346 Batch_id=3124 Accuracy=82.50: 100% 3125/3125 [00:46<00:00, 67.04it/s]

Test set: Average loss: 0.02744, Accuracy: 8471/10000 (84.71%)

EPOCH: 28
Loss=0.3816123604774475 Batch_id=3124 Accuracy=82.71: 100% 3125/3125 [00:48<00:00, 64.81it/s]

Test set: Average loss: 0.02697, Accuracy: 8526/10000 (85.26%)

EPOCH: 29
Loss=0.4151839017868042 Batch_id=3124 Accuracy=82.86: 100% 3125/3125 [00:46<00:00, 67.90it/s]

Test set: Average loss: 0.02792, Accuracy: 8491/10000 (84.91%)

EPOCH: 30
Loss=0.24157914519309998 Batch_id=3124 Accuracy=83.07: 100% 3125/3125 [00:46<00:00, 67.06it/s]

Test set: Average loss: 0.02733, Accuracy: 8493/10000 (84.93%)

EPOCH: 31
Loss=0.32522469758987427 Batch_id=3124 Accuracy=82.64: 100% 3125/3125 [00:47<00:00, 65.98it/s]

Test set: Average loss: 0.02712, Accuracy: 8531/10000 (85.31%)

EPOCH: 32
Loss=0.29400330781936646 Batch_id=3124 Accuracy=83.28: 100% 3125/3125 [00:46<00:00, 67.59it/s]

Test set: Average loss: 0.02711, Accuracy: 8530/10000 (85.30%)

EPOCH: 33
Loss=0.5165942907333374 Batch_id=3124 Accuracy=83.21: 100% 3125/3125 [00:47<00:00, 66.41it/s]

Test set: Average loss: 0.02721, Accuracy: 8520/10000 (85.20%)

EPOCH: 34
Loss=0.3541680574417114 Batch_id=3124 Accuracy=83.28: 100% 3125/3125 [00:47<00:00, 65.82it/s]

Test set: Average loss: 0.02644, Accuracy: 8579/10000 (85.79%)

EPOCH: 35
Loss=0.28175508975982666 Batch_id=3124 Accuracy=83.45: 100% 3125/3125 [00:47<00:00, 66.38it/s]

Test set: Average loss: 0.02662, Accuracy: 8546/10000 (85.46%)

EPOCH: 36
Loss=0.35980480909347534 Batch_id=3124 Accuracy=83.14: 100% 3125/3125 [00:45<00:00, 68.45it/s]

Test set: Average loss: 0.02635, Accuracy: 8575/10000 (85.75%)

EPOCH: 37
Loss=0.20571590960025787 Batch_id=3124 Accuracy=83.62: 100% 3125/3125 [00:47<00:00, 65.16it/s]

Test set: Average loss: 0.02723, Accuracy: 8539/10000 (85.39%)

EPOCH: 38
Loss=0.5773186087608337 Batch_id=3124 Accuracy=83.49: 100% 3125/3125 [00:46<00:00, 66.62it/s]

Test set: Average loss: 0.02643, Accuracy: 8511/10000 (85.11%)

EPOCH: 39
Loss=0.4408525228500366 Batch_id=3124 Accuracy=84.10: 100% 3125/3125 [00:46<00:00, 67.86it/s]

Test set: Average loss: 0.02606, Accuracy: 8572/10000 (85.72%)

EPOCH: 40
Loss=0.3040410876274109 Batch_id=3124 Accuracy=84.16: 100% 3125/3125 [00:48<00:00, 64.97it/s]

Test set: Average loss: 0.02565, Accuracy: 8601/10000 (86.01%)

EPOCH: 41
Loss=0.1682201623916626 Batch_id=3124 Accuracy=84.36: 100% 3125/3125 [00:46<00:00, 67.57it/s]

Test set: Average loss: 0.02594, Accuracy: 8572/10000 (85.72%)

EPOCH: 42
Loss=0.5251756310462952 Batch_id=3124 Accuracy=84.41: 100% 3125/3125 [00:46<00:00, 67.04it/s]

Test set: Average loss: 0.02533, Accuracy: 8606/10000 (86.06%)

EPOCH: 43
Loss=0.6298632621765137 Batch_id=3124 Accuracy=84.38: 100% 3125/3125 [00:47<00:00, 66.04it/s]

Test set: Average loss: 0.02551, Accuracy: 8613/10000 (86.13%)

EPOCH: 44
Loss=0.7830911874771118 Batch_id=3124 Accuracy=84.54: 100% 3125/3125 [00:47<00:00, 65.97it/s]

Test set: Average loss: 0.02541, Accuracy: 8619/10000 (86.19%)

EPOCH: 45
Loss=0.3663400113582611 Batch_id=3124 Accuracy=84.48: 100% 3125/3125 [00:47<00:00, 66.36it/s]

Test set: Average loss: 0.02545, Accuracy: 8607/10000 (86.07%)

EPOCH: 46
Loss=0.5790292024612427 Batch_id=3124 Accuracy=84.44: 100% 3125/3125 [00:47<00:00, 66.08it/s]

Test set: Average loss: 0.02536, Accuracy: 8614/10000 (86.14%)

EPOCH: 47
Loss=0.2978706955909729 Batch_id=3124 Accuracy=84.79: 100% 3125/3125 [00:48<00:00, 65.05it/s]

Test set: Average loss: 0.02536, Accuracy: 8595/10000 (85.95%)

EPOCH: 48
Loss=0.3022540807723999 Batch_id=3124 Accuracy=84.64: 100% 3125/3125 [00:46<00:00, 67.29it/s]

Test set: Average loss: 0.02541, Accuracy: 8606/10000 (86.06%)

EPOCH: 49
Loss=0.5252858400344849 Batch_id=3124 Accuracy=84.72: 100% 3125/3125 [00:46<00:00, 66.59it/s]

Test set: Average loss: 0.02522, Accuracy: 8617/10000 (86.17%)

EPOCH: 50
Loss=0.693017840385437 Batch_id=3124 Accuracy=85.02: 100% 3125/3125 [00:48<00:00, 64.62it/s]

Test set: Average loss: 0.02518, Accuracy: 8635/10000 (86.35%)

EPOCH: 51
Loss=0.9882620573043823 Batch_id=3124 Accuracy=85.03: 100% 3125/3125 [00:46<00:00, 67.15it/s]

Test set: Average loss: 0.02533, Accuracy: 8619/10000 (86.19%)

EPOCH: 52
Loss=0.2237468957901001 Batch_id=3124 Accuracy=85.07: 100% 3125/3125 [00:46<00:00, 67.08it/s]

Test set: Average loss: 0.02521, Accuracy: 8622/10000 (86.22%)

EPOCH: 53
Loss=0.256242036819458 Batch_id=3124 Accuracy=84.99: 100% 3125/3125 [00:46<00:00, 66.67it/s]

Test set: Average loss: 0.02504, Accuracy: 8633/10000 (86.33%)

EPOCH: 54
Loss=0.3043377101421356 Batch_id=3124 Accuracy=84.93: 100% 3125/3125 [00:46<00:00, 67.36it/s]

Test set: Average loss: 0.02519, Accuracy: 8635/10000 (86.35%)

EPOCH: 55
Loss=0.34830543398857117 Batch_id=3124 Accuracy=85.15: 100% 3125/3125 [00:46<00:00, 67.43it/s]

Test set: Average loss: 0.02516, Accuracy: 8639/10000 (86.39%)

EPOCH: 56
Loss=0.17848365008831024 Batch_id=3124 Accuracy=85.01: 100% 3125/3125 [00:47<00:00, 65.53it/s]

Test set: Average loss: 0.02486, Accuracy: 8662/10000 (86.62%)

EPOCH: 57
Loss=0.30877694487571716 Batch_id=3124 Accuracy=84.96: 100% 3125/3125 [00:46<00:00, 67.09it/s]

Test set: Average loss: 0.02508, Accuracy: 8658/10000 (86.58%)

EPOCH: 58
Loss=0.28278011083602905 Batch_id=3124 Accuracy=85.23: 100% 3125/3125 [00:46<00:00, 67.26it/s]

Test set: Average loss: 0.02517, Accuracy: 8648/10000 (86.48%)

EPOCH: 59
Loss=0.10036623477935791 Batch_id=3124 Accuracy=85.25: 100% 3125/3125 [00:47<00:00, 65.51it/s]

Test set: Average loss: 0.02493, Accuracy: 8656/10000 (86.56%)


--------------------------------------------------------------------------------------------------------------------------------

# Final Model Performance

Train set: Average loss: 0.02113, Accuracy: 44118/50000 (88.24%)
Test set: Average loss: 0.02493, Accuracy: 8656/10000 (86.56%)


--------------------------------------------------------------------------------------------------------------------------------

# Mis-classified Images

![misclassified_images](https://user-images.githubusercontent.com/21367838/217027997-fad38568-bd6d-4f05-94f1-e426b1752581.png)
