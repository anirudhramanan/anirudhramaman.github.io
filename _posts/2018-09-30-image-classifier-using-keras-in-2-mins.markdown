---
layout: post
title: What can you cook faster, instant noodles or a model using keras ?
date: '2018-09-30'
cover_image: '/content/images/2018/imageclassifierkeras/promo.jpg'
---

Hi everyone, this is my second post on Machine Learning. We will try to build an image classifier model from scratch using keras under 2 mins (yes, let's see if we can build a model before your noddles is ready :P)

Instead of digging into the code straight away, let's understand what keras is and how it can be useful.

### Keras

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. In simple words, it's nothing but a wrapper on top of popular machine learning libraries such as tensorflow, CNTK or Theano.

### When and why to use keras ?

Keras exposes simple API's which can be used to build a model without much deep understanding. As stated in the official site 'Keras is an API designed for human beings, not machines.

Keras is easy to learn, hence makes it easier for developers to build and try out different models tuned with different hyperparameters in quick time.

It is very useful in cases where:
1. There is need of building proof of concept for a given problem
2. To win a kaggle competition with few days left.

In nutsheel, before implementing or choosing which model suits best for a given problem statement, you can use keras to build and try out different types of models.

In this tutorial we will use keras to build a convolutional neural network to classify images (cats or dogs)

### Wait, What is Convolutional Neural Network ?

Convoluational Neural Network (CNN) is a deep feed-forward neural network which is mainly used to analyse and classify images. A CNN consists of an input, an output and n-numbers of hidden layers. A hidden layer can consist of convolutional layers, pooling layer and fully connected layers. 

A simple CNN is a sequence of layers, and every layer in the CNN transform one volume of activation to another using a function. 

![CNN](/content/images/2018/imageclassifierkeras/cnn.png)

Description of each layer:

* Input Layer: This layer is used to pass the raw pixel information of the image. Typically the raw pixel information looks like this [128 x 128 x 3] where 128 x 128 is the width and height of the image and 3 is the color channel (RGB)

* Convoluation Layer: This layer is used to calculate the outputs of neurons in that layer/region.

* RELU Layer: This is used to apply the activation functions to the output of previous layer.

* Pooling Layer: Pooling Layer performs downsampling and reducing the dimensionality of the given input image.

* Fully Connected Layer: This is used to compute the final class label for a given image. 

Want to read CNN in detail? [Read here](http://cs231n.github.io/convolutional-networks/)

### Implementation

Since we now have a rough idea of CNN and it's working, let's start writing code using keras. Don't remember to start your timer. 

#### Dataset

Download the cats and dogs image dataset from [kaggle](https://www.kaggle.com/c/dogs-vs-cats/data). Before implementing the model, we will use the ImageDataGenerator class of keras to prepare the data.

![Dataset](/content/images/2018/imageclassifierkeras/dog.jpg)

```python
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.preprocessing import image

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, 
                                   zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train', 
                                                 target_size = (64, 64), 
                                                 batch_size = 32, 
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test',
                                             target_size = (64, 64), 
                                             batch_size = 32, 
                                             class_mode = 'binary')
```

Start by importing the keras packages


```python
# import convolutional layer
from keras.layers import Conv2D
# used to build the model as a sequence
from keras.models import Sequential
# used for pooling
from keras.layers import MaxPooling2D
# used to flatten out the layers
from keras.layers import Flatten
# for fully connected layer
from keras.layers import Dense
```

A sequential model is a linear stack of layers. It takes in a list of layers, and stacks them in the order added. We will use this to add layers to the model


```python
# define a sequential model
model = Sequential()
```

We now add the convolutional layer. The convolutional layers (Conv2D function) takes in 3 arguments:

1. The number of filters
2. The filter dimensions ie 3x3
3. The input image size (first two values represent the width and height respectively and the third value is the color channel ie 3 for RGB, 1 for grey)


```python
# add the convolutional layer
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
```

Activation function defines the output of the node given an input or set of inputs. relu (Rectiier Linear Units are the most widely used activation functions right now which is mainly used to classify images)


```python
# add the relu activation function to the layer
model.add(Activation('relu'))
```

Once we have a convolutional layer, we add the max pooling layer. The objective of the pooling layer is to downsample the input images. 

The pooling layer resizes the dimensions of each input using the MAX operation.


```python
# add max pooling layer for dimensionality reduction
model.add(MaxPooling2D(pool_size = (2, 2)))
```

This model till now uses 1 convolutional layer with relu activation function followed by the max pooling layer. We can add more such layers to the model to improve the training, but for keeping in simple we will be using a single layer powered model.

We now use the Flatten function of keras to flatten the model ie we convert our 3D feature maps to 1D feature vectors.


```python
# flatten the model
model.add(Flatten())
```

The final steps includes adding a fully connected layer to the model. Dense is a module exposed by keras to add a FC layer to the model. It takes in the number of nodes/units that should be present in the hidden layers of the network. This is then followed by another activation function


```python
# add the FC layer along with relu activation function
model.add(Dense(64))
model.add(Activation('relu'))
```

Now to improve the accuracy and to decrease overfitting, we will use a dropout layer. The idea behind dropout is to drop some nodes so that the network can concentrate on other features and avoid overfitting.


```python
# add dropout to reduce data overfitting
model.add(Dropout(0.5))

# add another FC layer to the network
model.add(Dense(1))

# add activation function
model.add(Activation('sigmoid'))
```

Since we have now added all the layers to the model, it's time to compile the model.

`binary_crossentropy` to compute the loss function of the model, `adam` optimizer to calculate the gradient descent and `accuracy` metric to evaluate the model


```python
# compile the model with the following configuration
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

We now have a compiled model. We will use this to train on the image dataset that we had imported earlier.

1. training_set: the training set being used to train the model
2. steps_per_epoch: number of training steps per epoch
3. epochs: an epoch is where the model is trained on every single data in the training set.
4. validation_data: validation data set
5. validation_steps: validation steps


```python
model.fit_generator(training_set,
                    steps_per_epoch = 2000,
                    epochs = 50,
                    validation_data = test_set,
                    validation_steps = 2000)
```

### Predicting cat or a dog


```python
test_image = image.load_img('dataset/train/cats/cat_01.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
```

We have built an image classifier model in no time !! This shows how keras can be useful to build and test out different models in a quick time.

### References

* [Keras](https://keras.io)
* [CNN](http://cs231n.github.io/convolutional-networks/)