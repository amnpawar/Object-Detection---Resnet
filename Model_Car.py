from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from ResNet import ResNet
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,
	help="path dataset of input images")

'''
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
'''
model_train="trained_car.h5"
#TRAIN_PATH="F:/Machine Learning/Major Project/INRIAPersonIn/training_set"
TRAIN_PATH="D:/Projects/FinalCodes/car_dataset/car_training"
#TEST_PATH="F:/Machine Learning/Major Project/INRIAPersonIn/dataset"
TEST_PATH="D:/Projects/FinalCodes/car_dataset/car_testing"
NUM_EPOCHS = 7
BS = 32
args = vars(ap.parse_args())

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

 
# initialize the training generator
trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=32)


# initialize the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize our Keras implementation of ResNet model and compile it
model = ResNet.build(64, 64, 3, 2, (2, 2, 3),(32, 64, 128, 256), reg=0.0005)
opt = SGD(lr=1e-1, momentum=0.9, decay=1e-1 / NUM_EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train our Keras model
H = model.fit_generator(
	trainGen,
	steps_per_epoch=BS,
	validation_data=testGen,
	validation_steps=BS,
	epochs=NUM_EPOCHS)

print("[INFO] serializing network to '{}'...".format(model_train))
model.save(model_train)
print (os.path.abspath(model_train))
