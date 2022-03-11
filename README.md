# Intel_Image_Classification

## Exploratory Data Analysis + Data Visualization + Deep Learning Modelling 

### 1 - Abstract

In this project I made Exploratory Data Analysis, Data Visualisation and lastly Modelling. Intel Image Dataset contains 25 images in 3 different sets. Each example is 150x150 image and associated with __6__ labels(targets). After examining the dataset I used ImageDataGenerator for example rescaling images and increasing the artifical training and test datasets. In modelling part, with a InceptionResNetV2 and several other layers implemented. Model trained with __10__ Epochs for training the data. Also for long epochs time I implemented callback for time saving. Overally, model gives __0.9023__ accuracy. Furthermore with hyperparameter tuning model can give higher accuracy or using GPU for training it will reduce time and number of epochs can be increased.


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

Firstly, I checked data, which came three different dataset which are train, test and validation. Later I checked distribution of labels in datasets moreover I see all the classes(labels) equally distributed. So luckily I dont need to do Oversampling or Undersampling.

Number of images in Train Directory: 
* __Buildings:  2191__
* __Street:     2382__
* __Mountain:   2512__
* __Glacier:    2404__
* __Sea:        2274__
* __Forest:     2271__

### 4 - Data Preprocessing

For preparing datasets to the model I used ImageDataGenerator for rescaling which is __1/255.0__. Also I defined batch size __128__, and for creating artifical images I used rotating that takes maximum __60__ degree rotation. In the below code every parameters are visible with explanation. 
```
train_datagen = ImageDataGenerator(
        rescale=1/255.0,            #multiply the data by the value provided
        featurewise_center=True,    #create generator that centers pixel values
        rotation_range=60,          #maximum 60 degree random rotation
        width_shift_range=0.2,      #fraction of total width, if < 1, or pixels if >= 1.
        height_shift_range=0.2,     #fraction of total height, if < 1, or pixels if >= 1.
        shear_range=0.2,            #image disortion in axis
        fill_mode='nearest')        #fill the color with nearest neighborhood for example, aaaaaaaa|abcd|dddddddd

```

```
train_generator = train_datagen.flow_from_directory(
        train_path,
        shuffle=True,              #shuffling the order of the image that is being yielded
        target_size=(150,150),     #size of image
        batch_size=128,            #size of the batches of data 
        class_mode='categorical'   #predicting more than two classes so we will use categorical
    )
```

### 5 - Modelling 

I used pretrained InceptionResNetV2 model. The InceptionResNetV2 model is already trained more than 1 million images. Then I added these parameters over the pre-trained model.

```
x = tf.keras.layers.Dropout(0.2)(last_output)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(units=128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(units=6, activation='softmax')(x)
```
Finally I am compiling model according these parameters, I used RMSprop class and I gave learning rate 0.0002 and momentum 0.9( A scalar or a scalar Tensor). 
```
model.compile(
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0002, momentum=0.9, centered=True), 
        loss = ['categorical_crossentropy'], 
        metrics = ['accuracy']
    )
```
Additionally, because of higher epochs time I decided to implement Early Stopping class, which is stopping training when it doesnt improve anymore or extremly low.
```
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',          #which quantity will monitor
        min_delta=0.001,             #minimum change in the monitored quantity to qualify as an improvement
        patience=5,                  #number of epochs with no improvement after which training will be stopped
        verbose=1,                   #verbosity mode
        mode='auto',                 #the direction is automatically inferred from the name of the monitored quantity.
        baseline=None,               #baseline value for the monitored quantity
        restore_best_weights=True)]  #whether to restore model weights from the epoch with the best value of the monitored quantity
```

### 6 - Result & Future Work

As a result, my model gives overally good results which is the result of InceptionResNetV2 model.

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

Test Loss is __0.2682__

Test Accuracy is __0.9023__

For higher accuracy hyperparameter tuning can be implemented or using GPU for training it will reduce training time and with this time saving epochs number can be increased.


