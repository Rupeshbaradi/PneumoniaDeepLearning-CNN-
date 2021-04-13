# PneumoniaDeepLearning-CNN-
Detecting Pneumonia with Convolutional Neural Networks
# Identifying Pneumonia Symptoms with Deep Learning Techniques (CNN) 



 
## 1. Introduction 

Pneumonia is an infection of the pulmonary parenchyma. This lung inflammatory condition affects the small air sacs known as alveoli. Most common symptoms include a combination of fever, cough, chest pain, fatigue, and shortness of breath. There are many causes of the infection that leads to different types of pneumonia, the most common causes include viral infections such as bacteria, fungi, etc. Risk factors for pneumonia may increase the severity of the disease this is due to age, malnutrition or smoking. 

Along with physical examination, image diagnosis plays a major role in the detection of pneumonia. Chest radiographs are frequently used in diagnostic procedures and represent a fast, cost-effective alternative to map the nature, features, and extension of lung inflammations; opacity areas are commonly correlated to pneumonia-affected regions. 

 

Fig.1 - Normal chest X-Ray (left) and abnormal high-opacity chest X-ray (right) 

2. Objectives of this deep learning exercise 

The main objective of this project is to develop a Convolutional Neural Network (CNN) focused on binary classification. The model can differentiate if a patient´s lungs have symptoms of pneumonia based on X-Ray images and assign labels as “0 or 1”. The person is labelled “Normal = 0” if there are signs of healthy lungs or “Pneumonia = 1” otherwise. This is possible by passing an X-Ray Gray-scale image of the patient’s chest as input to the CNN. 

 

 

3. Description of the X-Ray image Dataset 

To train the model, we used the dataset which is available in Kaggle platform from Children’s Medical Centre which is formed by 5,863 chest X-Ray images in Anterior-Posterior (AP) view of children from one to five years old, as part of the patient's medical follow-up. It's indicated by the data provided that all low-quality or unreadable scans were previously removed.  

The dataset is organized into 3 folders (“Train”, “Test”, “Validation”) and contains subfolders for each image category (Opacity/Normal). 

 

Training observations: 4,192 (1,082 normal cases, 3,110 lung opacity cases). 

Validation observations: 1,040 (267 normal cases, 773 lung opacity cases). 

Testing observations: 624 (234 normal cases, 390 lung opacity cases). 

 

 

Fig.2 - Number of Training, Testing, and Validation observations 

 

The training and validation X-Ray image dataset has the same distribution of approximately 75% of images that are affected with pneumonia, the number of opacity cases corresponds to 62.5% of the total test dataset. 

4. Methodology 

 

4.1. Model Building 

For model building the necessary steps involved in data engineering were resizing and rescaling of the images of the whole dataset. 

 

 

4.1.1. Callbacks 

A callback is an object that performs an action within one or many stages of the training model process. It is particularly useful to reduce the training time by using the function Early Stopping from keras.callbacks. The metric that determines whether to stop the training is “validation_accuracy”, considering patience of 10 epochs. 

4.1.2. Data Augmentation and Metrics 

While training the model we used precision, recall and false negatives as metrics on each epoch to evaluate the model performance improvement on the validation data. To have new realistic input images, we performed data augmentation with the following parameters: shear_range = 0.2, rotation_range = 15, brightness_range = [0.9, 1.1] and zoom_range = 0.2. 

4.1.3. Cut Mix 

CutMix combines two random images from the training set. It slices a part of one image and pastes it onto another image. CutMix also mixes the labels of the two images proportional to the area of the size of the cut. Training with harder-to-identify images allows the model to improve in terms of classification ability. Using it in our first model and achieved better results. 

4.2. Loss function 

4.2.1. Categorical Cross-entropy 

When trying to solve a categorical classification problem, the most common loss function is Categorical Cross-entropy. Cross-entropy loss is specialized in measuring the performance of a classification model whose given output can be expressed as a probability float from 0 to 1. We decided to use the Categorical Cross- entropy keras class, “categorical_crossentropy” when compiling the first model. 

4.2.2. Focal Loss 

Focal Loss is used to assess the performance of the model based on entropy same as the Categorical Cross-entropy. It is a loss function that has been created specifically to solve some of the Cross-entropy problems. Some of them are image object detection, dataset imbalance and low feedback of hard-to-classify objects using alpha (countermeasure for imbalance) and gamma (importance of misclassified propagation) as hyperparameters which can be modified to calibrate the whole system. To apply this technique, we used the focal loss function with parameters alpha = 0.25 and gamma = 2. 

Fig.3 - Loss behaviour with different gamma values. 

 

Diagram

Description automatically generated 

4.3. Class Weights 

We set class weights (cw = {0: 0.26, 1: 0.74}) in the fit method to give more importance to the less represented class. It represents the percentage of each class in the training dataset. 

 

4.4. Transfer Learning 

Transfer Learning is a different approach for model improvement to take advantage of a pre-trained network that can act as a generic visualization model, specifically using its pre-learned features to ease the new model object identification process. 

Transfer Learning workflow can be exemplified as follows: 

Taking layers from a previously trained model. 

Freeze the information they contain during future training rounds.  

Add and train new layers on the new dataset. 

We used the pre-trained model VGG16. 

4.5. Optimizer 

As there are a considerable number of gradient descent optimizers available, the team decant on familiarity and well-known efficiency when choosing which ones to compare. RMSprop has proved its efficiency in previous exercises. On the other hand, Adam though not as familiar, presents the benefits of RMSprop plus the capability of quickly ignoring local minima. These two options are considered for the tuning of the model. After all optimizer definition is more empirical than mathematical. 

4.6. Tuning 

In the pursuit of improving the model, as an alternative to manual tuning the “Kerastuner” library is used. It works by performing an iterative loop that evaluates certain pre-specified hyperparameters combinations. We used the “Hyperband” algorithm, and the resultant model is compared with the manually tuned one. 

5. Network Architecture 

 

Fig.4 - Network architecture structure 

 

The network contains two convolutional layers with 32 filters and 16 filters of size (3, 3), two max pooling layers with filter of size (2, 2). The network also consists of a dense layer with 128 neurons and 64 neurons with ‘relu’ activation function and a dense layer with 2 neurons with ‘SoftMax’ activation. We do a Batch normalization after the second max pooling layer, drop out with probability 0.5 after the second dense layer and L2 regularization on the second convolutional layer. 

6. Results 

6.1. Model with Categorical Cross-Entropy 

We used the previously mentioned architecture with the Categorical Cross-entropy loss function, using CutMix in the train generator, which leads to the following results on the test data. 

 

Fig.5 - Confusion Matrix (left) and ROC curve (right). 

The micro-average ROC curve computes the metric taking into consideration each class, we have an area = 0.92, a good result which is validated by the confusion matrix.  

 

Fig.6 - Classification Report. 

We saw a better performance in the results as seen above. 

6.2. Model with class weights 

We used the same loss function as before mention taking into consideration the class weight. 

 

Fig.7 - Confusion Matrix (left) and ROC curve (right). 

The micro-average ROC curve area didn’t improve but we see a radical reduction in the number of false negatives, we decided that it would be better prioritizing the patients' health over the model false positive score/accuracy. 

 

Fig.8 - Classification Report. 

After considering the class weight, we saw a better performance in the results as seen above. 

6.3. Model with focal loss 

Giving the focal loss function the parameters: alpha = 0.25, and gamma = 1.5 we observed better results than the two already mentioned functions. 

 

Fig.9 - Confusion Matrix (left) and ROC curve (right). 

The micro-average ROC curve area improves, our model can identify normal cases better, and only 2 patients were predicted to have pneumonia when they didn’t. The recall of the normal cases improved as compared to before.  

 

Fig.10 - Classification Report. 

After using the focal loss, we saw a better performance in the results as seen above. 

 

 

 

6.4. Transfer Learning 

We used VGG16 model freezing the existing weights and did not include the top layers which we change by three dense layers.  

 

Fig.11 - Confusion Matrix (left) and ROC curve (right). 

 

Fig.12 - Classification Report. 

The results of the transfer learning model didn’t show any significant improvement when compared to the focal loss deep learning model. 

6.5. Model comparison  

Model 

Running time in minutes 

Number of epochs 

 

Simple categorical cross-entropy with CutMix 

78.56 

44 

 

Categorical cross-entropy with Class Weights 

32.17 

30 

 

Focal Loss  

76.52 

71 

 

VGG16 categorical cross-entropy 

127.64 

37 

Fig.13 - Table defining running time and Epoch number for each model. 

Though the Focal Loss model was not the fastest model to run, we consider its performance to be significant as it correctly classified a considerable number of normal cases while having a low number of false negatives. Due to class imbalance the previous models tend to predict the majority class better while our chosen model displayed the highest ROC value within the studied models. 

7. Conclusions 

In the report, we have portrayed how we have identified pneumonia using Convolutional neural networks (CNN) a deep learning concept. The CNN model allows the practitioner to identify the X-Ray image patterns to support the diagnosis of pneumonia to support their decisions. 

 

While developing the model we faced few obstacles such as the amount of time needed to train the network, which is more noticeable when training models with more available parameters, for example Transfer Learning, which does not always provide the best results. The second limitation is with the hardware capacity (memory and processing power in useful time) constraining the possible hyperparameters we could choose.  

Another problem was imbalanced data classes which we solved by using focal loss and class weight. While using Automatic Kerastuning, which uses random search methods to tune the model, it did not improve our results and for this reason we decided to manually tune the model which after adjusting the relevant parameters allowed us to achieve more satisfying results. 

Considerations for future model improvement 

All in all, the model provides the answers to our pneumonia imaging identification problem. In future work we could tune the model to allow to differentiate between distinct types of pneumonia and even other conditions (for example, tumours, inflammations and other immunological diseases) as these can also show high opacity regions in the X-rays. 

8. References 

https://www.kaggle.com/pcbreviglieri/pneumonia-xray-images 

https://towardsdatascience.com/increase-the-accuracy-of-your-cnn-by-following-these-5-tips-i-learned-from-the-kaggle-community-27227ad39554 

https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/ 

https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-018-0544-y 

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7345724/ 
https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html 

https://keras.io/api/losses/ 

https://medium.com/visionwizard/understanding-focal-loss-a-quick-read-b914422913e7 

https://medium.com/@kaitotally/adam-the-birthchild-of-adagrad-and-rmsprop-b5308b24b9cd 

https://www.tensorflow.org/tutorials/keras/keras_tuner 

https://keras.io/guides/transfer_learning/ 

 

 

 
