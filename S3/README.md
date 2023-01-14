# PART 1

## Explanation
### Forward Pass

Calculate output from each neuron by using the weights and inputs coming from previous layer.

### Loss Computation

Calculate loss as l2 loss by subtracting prediction from actual, squaring it and dividing by 1/2.

### Backward Pass 

Update weights w5, w6, w7, w8 by subtracting a factor of partial derivative of total loss w.r.t. w5, w6, w7, w8 respectively.
Derivative of loss w.r.t. w5, w6, w7, w8 can be calculated by chain rule. First we need to take derivative of loss w.r.t. activation out, then take derivative of activation output w.r.t pre-activation output, then take derivative of pre-activation output w.r.t. w5, w6, w7, w8 respectively.
1. ∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5
2. ∂E_total/∂w6 = ∂E1/∂w6 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w6
3. ∂E_total/∂w7 = ∂E2/∂w7 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o1/∂w7
4. ∂E_total/∂w8 = ∂E2/∂w8 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o1/∂w8 

Next weights w1, w2, w3 and w4 are updated. Chain of operation for calculation their derivative: take derivative of loss w.r.t activation output in output layer, derivative of activation output w.r.t. pre-activation output of output layer, derivative of pre-activation output of output layer w.r.t. activation output of hidden layer, derivative of activation output w.r.t. pre-activation output of hidden layer, derivative of pre-activation output of hidden layer w.r.t. weights and multiply all of them to get the partial derivatives. Subtract these partial derivatives by a factor to update the weights.

1. ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
2. ∂E_total/∂w2 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2
3. ∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3
4. ∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w4


<img width="524" alt="image" src="https://user-images.githubusercontent.com/21367838/212447689-15bd70a3-9bda-40db-8707-bd0707890b9f.png">

<img width="452" alt="image" src="https://user-images.githubusercontent.com/21367838/212447706-56bd6dcf-2800-448c-b72b-d351201d280b.png">

<img width="548" alt="image" src="https://user-images.githubusercontent.com/21367838/212447727-b8da4354-8912-45fc-9d93-2fa2e09444ea.png">

<img width="917" alt="image" src="https://user-images.githubusercontent.com/21367838/212447751-eaa4c58f-cf1f-4332-9fa1-c80c2bb6591a.png">


## Effect of Change of Learning Rate

Training becomes faster with increase in learning rate, while it slows down with decrease in learning rate.

LR = 0.1
<img width="322" alt="image" src="https://user-images.githubusercontent.com/21367838/212447804-71f10a19-636b-420a-8ea7-30dfa091dfbd.png">

LR = 0.2
<img width="316" alt="image" src="https://user-images.githubusercontent.com/21367838/212447862-87e9de4b-8d1b-4d91-9547-a8f33dbaced6.png">

LR = 0.5
<img width="316" alt="image" src="https://user-images.githubusercontent.com/21367838/212447876-383de12c-8642-41a7-b164-9329a8d6cc99.png">

LR = 0.8
<img width="314" alt="image" src="https://user-images.githubusercontent.com/21367838/212447891-fdba37eb-8fe1-4d7f-84e0-fe7da48026d0.png">

LR = 1.0
<img width="316" alt="image" src="https://user-images.githubusercontent.com/21367838/212447908-29805105-3cfb-46a6-a27d-b9e29deee90f.png">

LR = 2.0
<img width="316" alt="image" src="https://user-images.githubusercontent.com/21367838/212447919-d98fde34-4aec-4578-aed4-8045da1552ea.png">









# PART 2

## Model Architecture
1. convolution -> relu -> batch normalization
2. convolution -> relu -> batch normalization -> max pooling -> dropout
3. convolution -> relu -> batch normalization -> dropout
4. convolution -> relu -> batch normalization -> dropout
5. convolution -> relu -> batch normalization -> global average pooling -> dropout
6. fully connected -> relu -> batch normalization -> dropout
7. fully connected -> log softmax

<img width="383" alt="image" src="https://user-images.githubusercontent.com/21367838/212446357-5eb200fe-4af3-43fc-981c-001e6876f215.png">


## Training Logs

Test set: Average loss: 0.0242, Accuracy: 9921/10000 (99.21%)

loss=0.0018307261634618044 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 64.21it/s]

Test set: Average loss: 0.0232, Accuracy: 9935/10000 (99.35%)

loss=0.08837100118398666 batch_id=1874: 100%|██████████| 1875/1875 [00:28<00:00, 65.22it/s]

Test set: Average loss: 0.0205, Accuracy: 9939/10000 (99.39%)

loss=0.002903894754126668 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 63.67it/s]

Test set: Average loss: 0.0227, Accuracy: 9932/10000 (99.32%)

loss=0.04027240723371506 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 63.28it/s]

Test set: Average loss: 0.0200, Accuracy: 9936/10000 (99.36%)

loss=0.030976906418800354 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 62.97it/s]

Test set: Average loss: 0.0210, Accuracy: 9937/10000 (99.37%)

loss=0.06207926571369171 batch_id=1874: 100%|██████████| 1875/1875 [00:28<00:00, 64.73it/s]

Test set: Average loss: 0.0213, Accuracy: 9932/10000 (99.32%)

loss=0.025734011083841324 batch_id=1874: 100%|██████████| 1875/1875 [00:28<00:00, 64.86it/s]

Test set: Average loss: 0.0178, Accuracy: 9942/10000 (99.42%)

loss=0.0012785899452865124 batch_id=1874: 100%|██████████| 1875/1875 [00:28<00:00, 65.68it/s]

Test set: Average loss: 0.0185, Accuracy: 9947/10000 (99.47%)

loss=0.01287381537258625 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 62.87it/s]

Test set: Average loss: 0.0185, Accuracy: 9942/10000 (99.42%)

loss=0.004263182170689106 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 63.47it/s]

Test set: Average loss: 0.0209, Accuracy: 9944/10000 (99.44%)

loss=0.024548757821321487 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 62.51it/s]

Test set: Average loss: 0.0183, Accuracy: 9941/10000 (99.41%)

loss=0.005898096598684788 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 64.10it/s]

Test set: Average loss: 0.0187, Accuracy: 9942/10000 (99.42%)

loss=0.17401878535747528 batch_id=1874: 100%|██████████| 1875/1875 [00:29<00:00, 63.58it/s]

Test set: Average loss: 0.0158, Accuracy: 9951/10000 (99.51%)

loss=0.07753217220306396 batch_id=1874: 100%|██████████| 1875/1875 [00:28<00:00, 64.85it/s]

Test set: Average loss: 0.0169, Accuracy: 9950/10000 (99.50%)


## Final Model Performance

Train Data: Average loss: 0.0120, Accuracy: 59782/60000 (99.64%)

Test Data: Average loss: 0.0169, Accuracy: 9950/10000 (99.50%)

