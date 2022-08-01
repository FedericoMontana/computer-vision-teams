This is a simple proof of concept to manage Microsoft Teams using hands gesture (touchless)

I have create a couple of handy classes that abstract much of the complexity of the computer vision logic. Thre are several jupyter notebooks to run the PoC:

Notebooks starting with 01 creates the "data" to train a model. You can execute either of the following parts:
- data_cam creates the data by using your own cam, and you decide what gestures you want to include
- data_imgs creates the data by using images. I have used a public repository of hands images as an example

Notebooks starting with 02 train the model, using the data created in step 01. You can execute either of the following parts:
- trainer_tf trains the model using Tensoflow (this implements the logic from this repo: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
- trainer_other trains the model using other mechanism not related to neural networks. 

Notebooks 03 grabs the model created, and uses it with your webcam to command Teams.



Note that I have created just a few combinations to manage Teams, you can easily extend the Teams class to add more commands, and associate them with whatever gesture you want..