## Deep Learning Project ##

[image1]: ./images/1.png
[image2]: ./images/2.png
[image3]: ./images/3.png
[image4]: ./images/4.png
[image5]: ./images/5.png

### FCN Architecture
My final model consisted of the following layers:

| Layer         	|     Description	        | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x160x3 RGB image   | 
| Encoder_layer1, SeparableConv2D     	| Separable Convolution |
| Encoder_layer1, BatchNormazation		| Batch Normazation, 80x80x32 |
| Encoder_layer2, SeparableConv2D     	| Separable Convolution |
| Encoder_layer2, BatchNormazation		| Batch Normazation, 40x40x64 |
| Encoder_layer3, SeparableConv2D     	| Separable Convolution |
| Encoder_layer3, BatchNormazation		| Batch Normazation, 20x20x128 |
| Convolution 1x1 | 1x1 Convolution,  20x20x128 |
| Decoder_layer1, Bilinear Upsampling   | Bilinear Upsampling, 40x40x128|
| Decoder_layer1, Concatenation			| Concatenation with Encoder_layer2, 40x40x192 |
| Decoder_layer1, SeparableConv2D     	| Separable Convolution, 40x40x128|
| Decoder_layer1, BatchNormazation		| Batch Normazation, 40x40x128 |
| Decoder_layer2, Bilinear Upsampling   | Bilinear Upsampling, 80x80x128|
| Decoder_layer2, Concatenation			| Concatenation with Encoder_layer1, 80x80x160 |
| Decoder_layer2, SeparableConv2D     	| Separable Convolution, 80x80x64|
| Decoder_layer2, BatchNormazation		| Batch Normazation, 80x80x64 |
| Decoder_layer3, Bilinear Upsampling   | Bilinear Upsampling, 160x160x64|
| Decoder_layer3, SeparableConv2D     	| Separable Convolution, 160x160x32|
| Decoder_layer3, BatchNormazation		| Batch Normazation, 160x160x32 |
| Convolution 3x3 | 1x1 stride, same padding, 160x160x3 |
| Softmax				||

### Techniques in FCN
The big picture of FCN consists of 3 parts. 
	
1 The first part are Encoder Layers, which are trying to extract all kinds of features (color space,  spatial space ...) by downsamping in channels.

2 The second part are 1x1 convolutional layers, which are trying to refine all the features in specific channels.

3 The third part are decoder layers, which are trying to restore the origin size by upsampling without carrying the detail features, which are not needed by segmentation task.

![image1]

Let's deep into each of the part in my implementation.

#### Encoders

Encoder layers are trying to extract features in different channles. By channels, I mean some specific aspect which could describe the image, like "red value channel" or "whether or not has an edge channel". Of course, the network would not learn the exact feature channels as we expected. It would try to learn different feature channels so that the loss function is minimized.

In my implemention, the encoder pipilne consisted of 3 encoder layers. In every encoder_layer, separable convolutional layer, batch normalization layer was used to improve the performance.

**Separable Convolution**

Separable convolution is different from the regular convolution. Let's say we want to extract ```target_n``` channels in a convolutional layer. 

Traditional convolution would have to use n different kernel, each kernel convoluting with inputs and generating the target channel. In this way, parameter number would be ```target_n x 5 x 5 x input_n``` (```input_n``` is the number of channels of input and let's say we use kernels with size 5). 

Separable convolution would like to use single kernel to downsample from input, and then based on the downsampled result, ```target_n``` different ```1x1``` convolutional layers was usd to extract different features for different channels. In this way, parameter number would be ```5 x 5 x input_n + target_n x input_n```

By these changes, Separable convolution has much lesser parameters which still remains the functionality of convolution, which is downsampling and extracting features.

**Batch Normalization**

I used 3 Separable Convolution Layers in my implementation to extract enough features for future use. Output of previous layer would be the input of next layer. If the output of some layer was skewed, the network convergency would be slow because the input has no much difference, as a result of it, the gradient in back-propagation would be really small.

Batch Normalization was trying to focus on this problem, after every Separable Convolution Layer, before entering into the next layer, all the output would be normalized based on the mean and deviation (mean and deviation from training data).

#### 1x1 Convolution

Based on the output of encoder layers, ```1x1 Convolution``` layers was used to refine the spatials features in every channels. Fully connected features were not approviate here, because fully connected layers need to flatten output from encoders, which would distrupt the spatial features in every channel, which is against the goal.

#### Decoders

Decoder layers are trying to restore the origin size by upsampling without carrying the detail features, which are not needed by segmentation task. Several techniques were used in decoder, consisting of upsampling, skip-connection, separable convolution and batch normalization. Separable convolution and batch normalization were used to refine the spatial features after upsampling task, details about them have been discussed in the previous sub section. Let's look in to upsampling and skip-connection in details.

**Upsampling**

Transposed Convolution would be on way to do unsampling, transposed convolutional layers would be learned from training data. In my implementation, I used Bilinear Upsampler instead of Transposed Convolution to do the upsampling. The bilinear upsampling method does not contribute as a learnable layer, but it could speed up performance, because the network don't have to learn it.

Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, to estimate a new pixel intensity value. The weighted average is usually distance dependent.

![image2]

**Skip-Connection**

Skip-Connection was proved to be a great way to retain some of the finer details from the previous layers as we decode or upsample the layers to the original size. 

One way to carry this out is using an element-wise addition operation to add two layers. Because of the element-wise addtion, shape of the two layers would have to be same.

Concatenation was used in my implementation as another way of utilizing skip-connection. In this way, output from different layers would be concatenated along the depth (channel) axis, so it offers a bit of flexibility because the depth of the input layers need not match up.

![image3]

#### Generalization

The training samples contain only human images from simulator, it is unlikely to work well for following other objects like dog, cat or something else instead of human. In order to work well for those objects, extra training samples need to be collected for furture training of this model.

### Experimentations

1 The first set of hyper-parameters. From the training curves, the validation loss were bumping up and down, and the model was learning too fast in each step, learning rate should be reduce. 

	learning_rate = 0.01
	batch_size = 32
	num_epochs = 20
	steps_per_epoch = 256
	validation_steps = 64
	
![image4]

2 The second set of hyper-parameters. I reduced the learning rate to 0.001. With smaller ```learning_rate```, I reduced the ```batch_size``` to speed up the traning process. And from last experimentation, the network should be able to converge in less than 20 epochs, I also reduced steps_per_epoch to speed up training process.

	learning_rate = 0.001
	batch_size = 16
	num_epochs = 20
	steps_per_epoch = 200
	validation_steps = 50
	
![image5]

I did the check-points saving for every epoch in the traing process. From the training curve, the models started to overfit after the 9th epoch. I interrupted the training process, because the traning process was quite slow in my computer. I used ```weights.09.hdf5``` model to do the evaluation, the IOU metric was larger than 0.4.

```model_training.ipynb``` and ```model_training.html``` are both in the ```code``` directory, same is the ```h5``` model file named ```weights.09.hdf5```. 

Improvement could be made to improve the stability of traing process. Increasing the batch_size could be a good choice, increasing steps_per_epoch could be another one. They should both be able to make the model converge more smoothly.