---
title: Using transfer learning to train an image classifier
date: 2018-9-23
tags: [machine learning]
---

![Image](./main.jpeg)

Training a model from scratch requires large amount of training data and time in terms of computing power, even with a GPU. Why to train a model from scratch when there is a way to extract the lower level layers from an already pre-trained model, and reuse the same in the new model.

<b>Transfer learning</b> is reusing the already pre-trained model on the new training dataset, and is one of the most popular approaches in the machine learning domain. This turns out to be very useful when the training dataset and the computing power both are limited.

Transfer learning can:

- Can train a model with a smaller dataset,
- Improves generalization,
- Speeds up training.

Training a model using transfer learning could take approximately 40 mins to 1 hr (depending on the training dataset) of the CPU time on a normal laptop, as compared to hours of GPU time if built from scratch.

![Image Dataset](./dataset.png)

In this tutorial we will use the pre-trained inception-v3 model to build the dog breed classifier which will be trained on the standford dog breed dataset (dataset contains images of 120 different breeds from around the world, total of 20580 images)

---

### Note:

Explaining the inception-v3 model is not in the scope of this article, if interested you can read more about it [here](https://www.tensorflow.org/tutorials/images/image_recognition).

[Read](https://arxiv.org/abs/1512.00567) paper on inception

---

We will be using the retrain script (this is a part of tensorflow example sample) to retrain the model

### Understanding retrain.py

retrain script is used to retrain the model (inceptionv3 default model or mobilenet etc) with your own training dataset. By default, it uses the feature vectors computed by the inception v3 model.

Some of the important parameters that can used to tweak the model:

1. <b>-image_dir</b> : path of the labeled images. This is how the images directory will look like. The names of the folders are important here since these are used as the labels for each image inside them (The name of image file does not matter though)
![training Dataset](./training_set.png)

2. <b>-output_graph</b>: path to save the output ie the trained graph. By default these gets saved to the tmp directory in the root folder

3. <b>-intermediate_output_graphs_dir</b>: path to save the intermediate graphs.

4. <b>-intermediate_store_frequency</b>: number of steps to store intermediate graph.

5. <b>-output_labels</b>: path to store the output label (in this case labels will be the breed names)

6. <b>-summaries_dir</b>: this is used to visualise the graphs and histograms using tensorboard. Tensorboard is a tool to visualise the training graphs, learning and error rates

7. <b>-how_many_training_steps</b>: number of training steps. By default, this is 4000. You can also increase the number of training steps acc to the dataset that you use to train the model. 
   However, important thing to keep in mind is increasing the number of training steps does not guarantee an increase in the model accuracy since there can be cases where the dataset might be overfitting.

8. <b>-learning_rate</b>: the rate by which the model learns. Default value is 0.01. As implied, decreasing the rate will increase the time taken for the training process.

9. <b>-testing_percentage</b>: what portion of training set to be used as test set

10. <b>-validation_percentage</b>: what portion of training set to be used as validation set

11. <b>-eval_step_interval</b>: interval in which the training result is evaluated

12. <b>-train_batch_size</b>: number of images to train at a given point. default is 100, but this can be tweaked depending on the computing power.

13. <b>-test_batch_size</b>: number of images to test at a given time

14. <b>-validation_batch_size</b>: self explanatory

15. <b>-bottleneck_dir</b>: directory to store bottleneck values. We will cover bottleneck later in the post

16. <b>-final_tensor_name</b>: this is the name of the final output layer in the graph

17. <b>-tfhub_module</b>: which tensorflow module to use. by default it uses the inception v3 model, but this can be changed according to the use-case.

18. <b>-saved_model_dir</b>: directory to save the trained model

Some of the parameters like ( - random_brightness etc) which have not been mentioned here are mostly used to tweak the training dataset to train the model in different scenarios like an image in lower brightness, or a cropped image.

Now, since you have an idea of what type of parameters the script expects, let's see how it uses these to retrain the model.

1. Creates a list of training images (this uses the test and validation % flag ie how much part of training dataset to be used as test set and validation set)
2. Uses TensorFlow hub to fetch the pre-trained model. By default it uses the inception v3 model features to train the model.
3. After fetching the pre-trained model, it adds a new layer which is used for training on the new dataset.
4. Applies distortion to the images if any (ie if rotation, or background flag has been set)
5. Calculates bottleneck values for each image layer and caches them. (bottleneck has been explained in detail below)
6. Feeds the bottleneck values to the session and starts the training step.
7. Runs evaluation on validation and test set, and saves the model

This sounds simple, isn't it ?

We came across some new terms TensorFlow hub, and bottleneck, let's check what these means

<b>TensorFlow hub</b>: TensorFlow hub is a library for the publication, discovery, and consumption of reusable parts of machine learning models.

<b>Bottleneck</b>: the first phase of the script analyses the training images, and calculates the bottleneck values for each one of them. bottleneck is a term that is used to denote the layer just before the final output layer responsible for classification. <b>Why is this required ?</b> Every image can be used multiple times during the training phase, and calculating the layers before the bottleneck layer of image can eat up time. Since these lower layers are not modified and remains same through out the training phase, these are cached so that they are not calculated again which saves the training time.

Now since we have a fair understanding of how the retrain script works, let's run this script for our training dataset.

```

# run the retrain script and pass the training image directory. For simplicity, we will not be using other parameters as of now.
python retrain.py --image_dir Images

```

![Bottleneck](./bottleneck.png)

Once completed (which might take approximately 40–45 mins depending on the device being used for training), the retrained model will be saved to the directory provided (defaults to \tmp). This one step generates a trained model which can be used to classify images (dog breed in our case)

![Validation](./validation.png)

This model has a validation accuracy of 91% which is a very good accuracy rate for a model that has been trained in a limited time with limited amount of training set. We can now use the trained model to predict the breed for any given dog.

![DogBreed](./dog_breed.jpeg)

Do you know which breed the dog is ? No ? Shall we ask the model then ?

```
# test the model on a given image
# label_image is a simple script to load the image, and to read the tensor from the given file (https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py)

python label_image.py \
--graph=output_graph.pb \
--labels=output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=dog.jpg
```

![Predicted value](./predicted.png)

Wohooo!! You have built a dog breed classifier in no time :)

This shows how transfer learning helps in building a model with a good accuracy rate with limited amount of training data in a much quicker way.

[Github](https://github.com/anirudhramanan/transfer-learning-image-classifier) example for reference.

---

### When to use transfer learning ?

Transfer learning is generally used when the training dataset is not large enough, or if a model already exists which is trained on general features which you can use to retrain it on your custom dataset.

### Why is the accuracy low ?

One of the reasons for low accuracy is overfitting. This happens when the model instead of learning general features in the training dataset starts to memorise details which are not that important. One of the ways to tell if the model is overfitting is to look at the training and validation accuracy. If the model has a high training accuracy, but a low validation accuracy it is said to be overfitting the training set.

### References

[https://www.tensorflow.org/hub/](https://www.tensorflow.org/hub/)
[https://www.tensorflow.org/hub/tutorials/image_retraining](https://www.tensorflow.org/hub/tutorials/image_retraining)
[https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)
