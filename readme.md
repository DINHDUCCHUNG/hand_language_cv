## GROUP 13: Hand Language Detection

### Code Descripton:

#### video_capture.py:

- Get the video of each character such as A, B, C, .. then screen capture frames
- After that, convert these images to theshold image

#### train_model.py

- Load data set and split 80% data train and 20% data test
- Use `VGG16` pretrain model and some addition fully-connected layers with relu and the last layer with softmax
  activation function
- Save models into folder `models`

#### train_model_v2.py

- Load data set and perform data argument with flip horizontal and random rotate from -40deg to 40deg
- Split 80% data train and 20% data test
- Use `VGG16` pretrain model and some addition fully-connected layers with relu and the last layer with softmax
  activation function
- Save models into folder `models_v2`

#### train_model_resnet.py

- Load data set and perform data argument with flip horizontal and random rotate from -40deg to 40deg
- Split 80% data train and 20% data test
- Use `inception_resnet_v2` pretrain model and some addition fully-connected layers with relu and the last layer with
  softmax activation function
- Save models into folder `models_v3`

#### train_model_v4.py

- Load data set and perform data argument with flip horizontal and random rotate from -40deg to 40deg
- Split 80% data train and 20% data test
- Use `xception` pretrain model and some addition fully-connected layers with relu and the last layer with softmax
  activation function
- Save models into folder `models_v4`

#### detection.py

- Load model
- Open camere
- Key listeners with press `b` to capture background for open-cv convert frame to threshold, press `r` to reset
  background, press `q` to quit
- Calculate output and show the predict character on the screen

#### setup.txt

- store all libraries to install

#### folder `videos`

- contains video to frame capture to image

#### folder `data`

- contains all datasets