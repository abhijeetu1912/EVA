# PART 1

## Forward Pass

h1 = w1*i1 + w2*i2		\
h2 = w3*i1 + w4*i2		\
a_h1 = σ(h1) = 1/(1 + exp(-h1))		\
a_h2 = σ(h2) = 1/(1 + exp(-h2))		\
o1 = w5*a_h1 + w6*a_h2		\
o2 = w7*a_h1 + w8*a_h2		\
a_o1 = σ(o1)		\
a_o2 = σ(o20	

## Loss function

E_total = E1 + E2		\
E1 = 1/2 * (o1 - a_o1)^2		\
E2 = 1/2 * (o2 - a_o2)^2\

## Backward Pass

∂E_total/∂w5 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h1  \
∂E_total/∂w6 = (a_o1 - t1) * a_o1 * (1 - a_o1) * a_h2  \
∂E_total/∂w7 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h1  \
∂E_total/∂w8 = (a_o2 - t2) * a_o2 * (1 - a_o2) * a_h2  \

∂E_total/∂w1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - ah1) * i1  \
∂E_total/∂w2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - ah1) * i2  \
∂E_total/∂w3 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - ah2) * i1  \
∂E_total/∂w4 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - ah2) * i2




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

