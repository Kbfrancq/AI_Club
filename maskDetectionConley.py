"""

"""
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#data set paths
PATH =  "C:/Users/conle/Documents/AiClub/AIClub"
CATEG = ["jpgMask", "jpgFace"]

#Define shape 50 x50
SIZE = 250

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

data = []
labels= []

#Normalize data
#create training data method
for category in CATEG:
    #combine the main directory path with specific data folder
    path = os.path.join(PATH, category)
    #iterate through all images in the folder
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
        #simply loads image to size SIZExSIZE and save to omage
    	image = load_img(img_path, target_size=(SIZE, SIZE))
        #image to array instead of multidimensional
    	image = img_to_array(image)
        #need since we are using mobile nets
    	image = preprocess_input(image)

    	data.append(image)
        #with mask 0 and without 1
    	labels.append(category)





#one hot encoding on the labels
# perform one-hot encoding on the labels  dont thinkg this is necessary
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
#data needs to be in numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

#take data out of training data and make test data

print("spilt trianing data ")
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
#print(len(trainX, " ", testX, " ", trainY, " ", testY))
print("shuffle training data")

#shuffle the list of data
#np.random.shuffle(training_data)
#np.random.shuffle(test)


# construct the training image generator for data augmentation
#this creates more data imag, basically by distorting the current data. (rotating, zoomin, etc)
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#// build model
print("building model")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(SIZE, SIZE, 3)))
                    #3 because RGB

model = keras.Sequential(); 

model.add(baseModel)

model.add(keras.layers.AveragePooling2D(pool_size=(7,7)))

model.add(keras.layers.Flatten())
#relu is used for non-linear use cases (for images)
model.add(keras.layers.Dense(128, activation='relu'))
#Dropout-->avoid overfitting
model.add(keras.layers.Dropout(.5))
#2 because there are two options, softmax for binary answer as well
model.add(keras.layers.Dense(2, activation='softmax'))

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#opt is used for images, track accuracy
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train the head of the network
print("[INFO] training head...")
#aug.flow adds extra data
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# save the model for later uses in other programs
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")
model.save('model.h5')
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")






