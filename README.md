<p align="center">
<a href="https://dscommunity.in">
	<img src="https://github.com/Data-Science-Community-SRM/template/blob/master/Header.png?raw=true" width=80%/>
</a>
	<h2 align="center"> < Face-Emotion-Recognition > </h2>
	<h4 align="center"> < Real-time facial emotions recognition model deployed in a website using Heroku > <h4>
</p>


##Dataset construction and preprocessing 
<So Basically the data is from -> https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset we didn’t use the complete dataset as the data was imbalanced we picked out only 4 classes and we manually had to go through all the images in order to clean them and we finally split them into a ratio of 80:10:10 train:test:valid  respectively. So the images are 48x48 gray scale images cropped to face using haarcaascades. 28275 train 3530 train 3532 validation these where the exact no. of images taken from kaggle but the number of images used to train will vary as we have used image generator and manual cleaning was also done. For the parameters used for image data generator u can check the model.ipynb.>
##Model construction 
<VGG16 or Visual Geometry Group which is a 16-layer model with a feature transformer to give essential features to predict output class of dataset, and classifier Dense layers connected to blocks of Convolutional and MaxPooling2D layers having output neurons same in number as the classes in our dataset, is a famous Transfer learning model trained on 1 million ImageNet images. Out of several transfer techniques like ResNet50 and DenseNet tried, VGG16 trained till ‘block5_conv1’ layer gave the best, 75% generalization accuracy. With a batch size of 64 and input shape 48*48, all images were flipped. Its top includes fully connected Dense layers, Dropout and Pooling2D layer with RMSProp optimizer warming up the model for 30 epochs, followed by blocks of Convolutional layers with layers.trainable set to True so that the top layers of the frozen model base are unfrozen and the convolutional weights learned by model by fitting for 25 more epochs on our dataset is used. The total trainable parameters in the model were 11,208,388. All the layers of the pre-trained VGG16 model are named. VGG16 final convolved layer comes down to 3*3 convolutions, the convolutional blocks are left are trained till ‘block5_conv1’ layer, with Adam optimizer and learning rate of 1e-4, using Keras with Tensorflow backend.>

## Instructions to run

* Pre-requisites:
	-  < insert pre-requisite >
	-  < insert pre-requisite >

* < directions to install > 
```bash
< insert code >
```

* < directions to execute >

```bash
< insert code >
```

## Contributors

<table>
<tr align="center">


<td>

John Doe

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
<a href = "https://github.com/person2"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/person2">
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
<a href = "https://github.com/person3"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/person3">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
</tr>
  </table>
  
## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

<p align="center">
	Made with :heart: by <a href="https://dscommunity.in">DS Community SRM</a>
</p>

