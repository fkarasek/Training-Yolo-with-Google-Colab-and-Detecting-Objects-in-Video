# Training-Yolo-with-Google-Colab-and-Detecting-Objects-in-Video

Hello, it is Francis.

I've been practicing using [YOLO](https://pjreddie.com/darknet/yolo/) for object detection and classification. The ultimate 
goal was to learn how to train the pretrained model using my own data. Attempting to train using my Macbook CPU would be 
prohibitively long. So the best option would be to use the GPU in google colab to train the model. And then download the 
weights onto our Macbook and use them to classify objects in a video. I will show you how I did it.

The steps taken are:

1. [ Setup Darknet on our personal computer ](#setup)
2. [ Download, clean, and process data ](#clean)
3. [ Upload to drive and mount to Colab - train ](#train)
4. [ Use new weights to classify objects ](#detect)

<a name="setup"></a>
## 1. Setup Darknet

Download darknet
```
$ git clone https://github.com/pjreddie/darknet.git
```
```
$ cd darknet
$ make
```
Then download the weights into the darknet folder
```
$ wget https://pjreddie.com/media/files/darknet53.conv.74
```

These are our pretrained convolutional neural network weights. They have been pretrained using ImageNet.
The next step in our setup is to change the makefile and yolov3.cfg files. For the makefile, simply just change 
GPU = 0 to GPU = 1, since we will be using a GPU to train. For the .cfg file, change the "classes=1" lines to the number
of classes of objects you wish to train your model with. In our case, we used "classes=7". Finally, we have to change some 
of the filters. I noted which ones I changed in a comment in the uploaded cfg file. (filters = (num/3) * (5+classes)).

<a name="clean"></a>
## 2. Download and Clean Data

The images, annotations, mp4, as well as the output video and files can be found [here.](https://drive.google.com/drive/folders/11gPddDkQqm7pukpkgAksrHGJUR2QluEf?usp=sharing)
When we examine the dataset more closely, we see that the annotations are not in the form that we need them to be in. 
Furthermore, there appears to be 513 images and only 360 annotation files. This means we must filter out the images that 
have associated annotations. 

The annotation format for YOLO needs to be .txt-file for each .jpg-image-file - in the same directory and with 
the same name, but with .txt-extension, and put to file: object number and object coordinates on this image, 
for each object in new line: '<object-class> <x> <y> <width> <height>'

The coordinates of the object must be normalized as follows:
Object-class: an integer between 0 and n-1 classes corresponding to the classes.names file of object names. 

x: x-coordinate of object in pixels / pixel width of entire image

y: y-coordinate of object in pixels / pixel height of entire image

width: object width in pixels / pixel width of entire image

height: object height in pixels / pixel height of entire image

An example image1.jpg would have an associated annotation image1.txt with the following format:

```
0 0.5444444444444444 0.1287037037037037 0.016666666666666666 0.028703703703703703
0 0.4701388888888889 0.12407407407407407 0.014583333333333334 0.027777777777777776
1 0.6180555555555556 0.21203703703703702 0.02013888888888889 0.053703703703703705
6 0.7548611111111111 0.2064814814814815 0.0125 0.05
3 0.4395833333333333 0.09444444444444444 0.021527777777777778 0.05185185185185185
```

I have provided an example python notebook "json_annotations.ipynb" if using my images and annotations.
Simply just change the filename paths to your own. Once the data is in good form, create a folder in your darknet folder titled
"traffic". Within this folder, put your folders titled "images" and "labels" with your data. In ddition, you need to copy 
the .cfg file and "Train_test_split.py" file into the traffic folder and also make two new files: darknet.data and classes.names. 
I have provided both of these files. classes.names contains the names of the objects you wish to classify. Each object name 
must correspond to the object-class integer in the labels files. For example: "car" in line one would correspond to a 0 in 
the object-class. 

The next file, darknet.data provides the number of classes of objects as well as directory pathways to the train-test split 
files, classes.names file and a backup folder to store our trained weights. Create this "traffic_train_weights" folder in 
the darknet folder. Also create a blank train_log.txt file in the darknet folder. This is where the training progress will be 
stored. The sample_train.txt and sample_test.txt files will be created once you run the train_test_split.py file
in Google Colab. These two txt files must contain the absolute pathways to the images in google drive, and so the .py file
cannot be run until you mount your drive into Colab. 

<a name="train"></a>
## 3. Upload and Mount to Colab

Upload the entire darknet folder into your Google Drive. Once finished, open the "Train_Yolo_Colab.ipynb" file in Google
Colab and select "Change runtime type" from "Runtime" tab, and select Python 3 and GPU. Now, just run the provided script to
setup the necessary environment and then start training!

Important: Google Colab only allows for 12 hours of use on the GPU. This is usually not a problem since the weights and 
train_log progress will automatically be saved in your google drive. In addition, you'll see a yolov3.backup file produced.
This backup weight can be remounted into colab for further training if the user wants to do more batches. In general, the more
classes there are, the more training we have to do. 

<a name="detect"></a>
## 4. Acquire New Weights and Detect!

Before we go ahead and detect some objects, we need to have openCV installed. OpenCV will deconstruct our video file frame by 
frame and detect objects in each of these frames. If you don't have it yet, just do a pip install in the command line. 

Download the train_log.txt file and the yolov3.backup file. Put these in the darknet folder on your computer. 

Now we must ask, did we do enough training? How can we tell? Before we use our weights, we would like to visualize the 
performance of them. We do this by plotting the batch number vs the average loss during the training. As the loss converges 
to a minimum, we know we have trained enough. We will execute the file Plot_train_loss.py in the command line:

```
$ python3 Plot_train_loss.py /Full/path/to/train_log.txt
```

This generates the following plot:
![alt text](https://github.com/fkarasek/Training-Yolo-with-Google-Colab-and-Detecting-Objects-in-Video/blob/master/training_loss_plot.png)

We see that the plot converges pretty soon - somewhere around the 125th batch. This is atypical but probably due to the 
small amount of data that we used. 

Now that we have determined that our training is complete, let's detect some obects in our video. Simply just refer to the
final_yolo.py file. Copy it into the darknet directory and in your command line run the following script:

```
$ python3 final_yolo.py --video=sample.mp4
```

The video output that I got can be seen [here.](https://drive.google.com/drive/folders/11gPddDkQqm7pukpkgAksrHGJUR2QluEf?usp=sharing)
It isn't perfect, but it is my first attempt at training. I had to make some very slight corrections to the alignent of 
the bounding boxes. This discrepancy is likely due to either a floating point error or a miscalculation in transferring the 
data to YOLO format. When converting the JSON annotations, I assumed that the x and y coordinates were the exact centers of
each object. 
Also, when OpenCV takes in an input, it must resize the pixel lengths and widths. The resizing of the images must be in 
integer numbers of pixels, so some leftover conversion beyond the decimal point is unaccounted for. This could lead to some
slight discrepancies when aligning the bounding boxes to these resized images. 
To improve our weights and confidence interval, I would suggest gathering more images and annotations to train via ImageNet. 
