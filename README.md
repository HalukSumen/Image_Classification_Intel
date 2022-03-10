# Intel_Image_Classification

## Exploratory Data Analysis + Data Visualization + Deep Learning Modelling 

### 1 - Abstract

In this project I made Exploratory Data Analysis, Data Visualisation and lastly Modelling. Intel Image Dataset contains 25 images in 3 different sets. Each example is 150x150 image and associated with __6__ labels(targets). #After examining the dataset I made data preprocessing for reshaping columns from __784__ to __(28,28,1)__ and save the target feature as a seperate vector. In modelling part, with a sequential model with multiple convolution layers with __50__ Epochs for training the data. For prediction overfitting and underfitting I adjust Dropout Layers. Overally, model gives __0.9236__ accuracy. Furthermore with Data augmentation and/or incresing data size can be helpful for taking better result. 


### 2 - Data
Intel Image Dataset contains __25,000__ examples,train set of __14,000__ test set of __3,000__ examples and validation set of __7,000__ . Each example is a __150x150__  image, associated with a label from 6 labels.

Each training and test example is assigned to one of the following labels:

* __0 Buildings__
* __1 Forest__
* __2 Glacier__
* __3 Mountain__
* __4 Sea__
* __5 Street__


<p align="center">
  <img width="500" height="300" src="https://github.com/HalukSumen/Image_Classification_Intel/blob/main/images/trainset.png">
</p>
<p align="center">
     <b>Train Dataset Example</b>
</p>

<p align="center">
  <img width="500" height="300" src="https://github.com/HalukSumen/Image_Classification_Intel/blob/main/images/testset.png">
</p>
<p align="center">
   <b>Test Dataset Example</b>
</p>

### 3 - Exploratory Data Analysis

Firslty, I checked data, which came three different dataset which are train, test and validation. Later I checked distribution of labels in datasets moreover I see all the classes(labels) equally distributed. So luckily I dont need to do Oversampling or Undersampling.

Number of images in Train Directory: 
* __Buildings:  2191__
* __Street:     2382__
* __Mountain:   2512__
* __Glacier:    2404__
* __Sea:        2274__
* __Forest:     2271__

### 4 - Data Preprocessing

For preparing datasets to the model I used ImageDataGenerator for rescaling which is __1/255.0__. Also I defined batch size __128__ and for creating artifical images I used rotating that takes maximum __60__ degree rotation.



processing which is reshaping columns from (784) to (28,28,1), and for seperate vector I save label feature then process test and train data. After that I split train set into train and validation dataset. Validation set contains %30 of original train dataset and split will be 0.7/0.03. Later this process I controlled distribution of labels in train dataset and validation dataset.

<p align="center">
  <img width="750" height="500" src="https://github.com/HalukSumen/FashionMnist/blob/main/images/number%20of%20items%20in%20each%20class%20in%20dataset.png">
</p>
<p align="center">
   <b>Number of Items in Each Class in Dataset</b>
</p>

<p align="center">
  <img width="750" height="500" src="https://github.com/HalukSumen/FashionMnist/blob/main/images/number%20of%20items%20in%20each%20class%20in%20validation%20dataset.png">
</p>
<p align="center">
   <b>Number of Items in Each Class in Validation Dataset</b>
</p>

### 5 - Modelling 

I used Sequential model. The sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. Then I add Conv2D layer, MaxPooling2D, Flatten and Dense. For each layer I used these parameters.

__1.Conv2D__
* filters = 32
* kernel_size = (3,3)
* activation function = relu 
* kernel_initializer = normal
* input_shape = (28,28,1)

__2.MaxPooling2D__
* pool_size = (2,2)


__3.Conv2D__
* filters = 64
* kernel_size = (3,3)
* activation function = relu 

__4.Flatten__


A flatten operation on a tensor reshapes the tensor to have the shape that is equal to the number of elements contained in tensor non including the batch dimension and doesnt need any parameters.

__5.Dense__


In first Dense Layer,
* units = 128
* activation function = relu


In second Dense Layer,
* units = 10
* activation function = softmax

Finally I am compiling model according these parameters,

* loss = categorical cross entrophy
* optimizer = adam
* metrics = accuracy

### 6 - Result & Future Work

As a result, my model gives overally good results. 

<p align="center">
  <img width="750" height="500" src="https://github.com/HalukSumen/Image_Classification_Intel/blob/main/images/training_validation_acc.png">
</p>
<p align="center">
   <b>Accuracy of the Model</b>
</p>

<p align="center">
  <img width="750" height="500" src="https://github.com/HalukSumen/Image_Classification_Intel/blob/main/images/training_validation_loss.png">
</p>
<p align="center">
   <b>Loss of the Model</b>
</p>

Test Loss is __0.2166__


Test Accuracy is __0.9236__


<p align="center">
  <img width="500" height="400" src="https://github.com/HalukSumen/FashionMnist/blob/main/images/classification%20report.png">
</p>
<p align="center">
   <b>Classification Report</b>
</p>
<p align="center">
  <img width="500" height="500" src="https://github.com/HalukSumen/FashionMnist/blob/main/images/true%20prediction.png">
</p>
<p align="center">
   <b>Correctly Predicted Items</b>
</p>
<p align="center">
  <img width="500" height="500" src="https://github.com/HalukSumen/FashionMnist/blob/main/images/false%20prediction.png">
</p>
<p align="center">
   <b>Falsely Predicted Items</b>
</p>




The Best accuracy is for Trousers(Class 1), Sandals(Class 5) with __0.99__ and worst accuracy is Shirt(Class 6) with __0.78__.


The Best recall is for Trousers(Class 1), with __0.99__ and worst recall is Shirt(Class 6) with __0.79__


The Best F-1 Score is for Trousers(Class 1) with __0.99__ and worst F-1 Score is Shirt(Class 6) with __0.78__


For better results, data augmentation can be implemented or data size can be expandable. 

