# Human Face Feature Classification
This repository allows you train a model on human features (eyes, nose, mouth, ...). It allows to you to test your model on some pictures. You can also run you model on you webcam or any video from the web. 

## Table of Contents
1. [Imports](#Imports) 
2. [What you need](#What-you-need)
3. [Results](#Results)
4. [Additional Notes](#Additional-Notes)

## Imports
```
tensorflow                2.2.0
opencv-python             4.3.0.36
numpy                     1.18.5
scipy                     1.5.0
matplotlib                3.2.1
```

## What you need
- A dataset with the following dimentions HEIGHT, WIDTH = (218, 178), with feature annotations
    - I used the following https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
    - The dataset goes in ./humans/img_align_celeba
    - The features goes in ./list_landmarks_align_celeba.txt
    - Note that the data set is pretty big (5 million celeb images), I only used around 5000 to train the model

- ./haarcascade_frontalface_default.xml, you can find this online just search for it
- If you are going to train the model on video data you need ./videoData directory