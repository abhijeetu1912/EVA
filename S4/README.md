# STEPS
1. Build the basic skeleton
2. Build lighter model
3. Add batch normalization & dropout
4. Add data augmentation
5. Add Learning rate scheduler

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


