# Face Recognition

Face Recognizer identifies the face of an individual by their name with the help of their facial features.
Face Recognizer uses deep learning algorithms to compare a live capture or digital image with the stored faceprints(also known as datasets) to verify identity.
The algorithm used for detection is haar cascade by Paul Viola and Michael Jones and for recognition is the LBPH Face Recognition algorithm.

### Haar Cascade for face detection

The technique trains a cascade function (boxes of shapes) that appears in images with faces and learns the general pattern of a face through the change in colours/shadows in the image. In the original paper, the author claims to have achieved 95% accuracy in face detection. You can find a detailed explanation in the OpenCV documentation.

References:
* https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
* https://towardsdatascience.com/face-detection-haar-cascade-vs-mtcnn-414c97cf3388

### LBPH Algorithm for face recognition

Local Binary Pattern (LBP) is a simple yet very efficient texture operator which labels the pixels of an image by thresholding the neighbourhood of each pixel and considers the result as a binary number. Using the LBP combined with histograms we can represent the face images with a simple data vector. 

References:
* https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b

## Prerequisites
1. [OpenCV 3.x](https://www.python.org/downloads/)
2. [Python 3](https://pypi.org/project/opencv-python/)
3. [Numpy](https://pypi.org/project/numpy/)

## Installing and Running
- Download the project as a zip file and unzip it.
- Download all the prerequisite libraries to run the program. 
  ```
  pip install opencv-python 
  pip install opencv-contrib-python
  pip install numpy
  ```
- Run 'createData.py'. This python file will ask for an ID(type any integer value) to enter and will help create a dataset by turning on the camera and capturing the images. You can also run the command ``` python createData.py``` in your CLI.

    ![Screenshot (3)](https://user-images.githubusercontent.com/72027411/210846974-9bca4b02-c74a-4d43-9734-7402a799e55c.png)

- Now comes training the dataset. run ```python trainData.py``` this will also create a trainData.yml file that will help in configuration while recognising someone.

    ![Screenshot (2)](https://user-images.githubusercontent.com/72027411/210847168-771e5e17-bf38-45f2-a9cb-7dc6fbeb13ae.png)
  
- Finally we will run ```python recogniseData.py``` this will turn on your camera and will recognise the face.

    ![Screenshot 2023-01-05 170316](https://user-images.githubusercontent.com/72027411/210855944-471c8444-c33c-43cc-9a34-fb7d18e77fa7.png)
