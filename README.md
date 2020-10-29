<p align="center">
<a href="https://dscommunity.in">
	<img src="https://github.com/Data-Science-Community-SRM/template/blob/master/Header.png?raw=true" width=80%/>
</a>
	<h2 align="center"> Face-Emotion-Recognition </h2>
	<h4 align="center"> Real-time facial emotions recognition model <h4>
</p>


## How we did dataset construction and preprocessing 
<p>
Data Preprocessing Steps:

1) Resizing the images 
2) Manually cleaning the datasets to remove incorrect expressions
3) Spliting the data into train ,validation and test(80:10:10)
4) Applying image augmentation
5) Haar Cascades to crop out only faces from the images from live feed while getting real time predictions

So Basically the data is from -> https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset we didn’t use the complete dataset as the data was imbalanced we picked out only 4 classes and we manually had to go through all the images in order to clean them and we finally split them into a ratio of 80:10:10 train:test:valid  respectively. So the images are 48x48 gray scale images cropped to face using haarcaascades. 28275 train 3530 train 3532 validation these where the exact no. of images taken from kaggle but the number of images used to train will vary as we have used image generator and manual cleaning was also done. For the parameters used for image data generator you can check the [model.ipynb](https://github.com/Data-Science-Community-SRM/Face-Emotion-Recognition/blob/master/VGG16_Modified/vgg16-modified-fer.ipynb).</p>


## Model construction 
<p>Deep Learning Model
After manually pre-processing the dataset, by deleting duplicates and wrongly classified images, we come to the part where we use concepts of Convolutional Neural Network to build and train a model that predicts the facial emotion of any person. The four face emotions are: Happy, Sad, Neutral and Angry. 

The data is split in training and validation sets: 80% Training, 20% Validation. The data is then augmented accordingly using ImageDataGenerator.

VGG-16 was used as the transfer model. After importing it, we set layers.trainable as False, print the model summary and select a favorable output layer, in this case, ‘block5_conv1’. This freezes the transfer learning model so that we can pre-train or ‘warm up’ the layers of our sequential model on the given data before starting the actual training. This helps the sequential model to adjust weights by training on a lower learning rate.

Setting the Hyper Parameters and constants (Only the best parameters are displayed below):
•	Batch size : 64

•	Image Size : 48 x 48 x 3

•	Optimizers :
o	RMS Prop (Pre-Train)
o	Adam

•	Learning Rate : 
o	Lr1 = 1e-5 (Pre-Train)
o	Lr2 = 1e-4

•	Epochs 
o	Epochs 1 = 30 (Pre-Train)
o	Epochs 2 = 25

•	Loss : Categorical Crossentropy

Defining the Model: Using Sequential, the layers in the model are as follows:
•	GlobalAveragePooling2D
•	Flatten
•	Dense (256, activation: ‘relu’)
•	Dropout (0.4)
•	Dense (128, activation: ‘relu’)
•	Dropout (0.2)
•	Dense (4, activation: ‘softmax’)
 The pre-training is done by using RMSProp at learning rate: 1e-5 and for 30 epochs.
After pre-training, we set layers.trainable as True for the whole model. Now the actual training will start. It is done by taking Adam optimizer at learning rate: 1e-4 for 25 epochs.
We were able to achieve a decent validation accuracy of 75% and an accuracy of 85%.
All the metrics observed during the model training are displayed on one plot: 
 


</p>


## Instructions to run

* Requirements:

-  The software requirements are listed below:

• pillow

• numpy==1.16.0

• opencv-python-headless==4.2.0.32

• streamlit

• tensorflow

* Download the zip file from our repository and unzip it at your desired location.

* Enter the following line of code in your teminal to run the streamlit script

```bash
<Streamlit run app.py >
```

## Contributors

<table>
<tr align="center">


<td>
Rakesh

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/template/blob/master/logo-light.png?raw=true"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/person1"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/person1">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td>

Stuti Sehgal

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/template/blob/master/logo-light.png?raw=true"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/stutisehgal"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/stutisehgal">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>



<td>

Bhavya 

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/template/blob/master/logo-light.png?raw=true"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/bhavya1600"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/bhavya-chhabra-1600">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

 <td>

Shubhangi Soni

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/template/blob/master/logo-light.png?raw=true"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/ShubhangiSoni1603"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/shubhangi-soni-165621188/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
 <td>

Sheel

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/template/blob/master/logo-light.png?raw=true"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/sheel1206"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/sheel1206">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
 <td>

Soumya

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/template/blob/master/logo-light.png?raw=true"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/person3"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/person3">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
<td>

Krish

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/template/blob/master/logo-light.png?raw=true"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/krishsabnani"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/krish-sabnani-15073a169">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
<td>

vignesh

<p align="center">
<img src = "https://github.com/Data-Science-Community-SRM/template/blob/master/logo-light.png?raw=true"  height="120" alt="Your Name Here (Insert Your Image Link In Src">
</p>
<p align="center">
<a href = "https://github.com/vignesh772"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/vignesh-v-1421b7191/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

  
## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

<p align="center">
	Made with :heart: by <a href="https://dscommunity.in">DS Community SRM</a>
</p>

