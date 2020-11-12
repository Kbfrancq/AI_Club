import numpy as np
import random
import os
import cv2
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

#data set paths
PATH =  "C:/Users/conle/Documents/AiClub/AI_Club_20-21/small_data"
CATEG = ["withbig", "withoutBig"]

#Define shape 50 x50
SIZE = 100

training_data = []

#Normalize data
#create training data method
def createTD():
    #iterate through all examples
    for category in CATEG:
        path = os.path.join(PATH, category)#makes path to both folders
        class_num = CATEG.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)                
                new_array = cv2.resize(img_array, (SIZE,SIZE))
                
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
#normalize method      
def norm(tester):
    new= []
    for i in range(len(tester)):
        tempx= []
        for x in range(len(tester[i])):
            tempy = []
            for y in range(len(tester[i][x])):
                tempy.append(float(tester[i][x][y]/255))
            tempx.append(tempy)
        new.append(tempx)
    return new


#create the training data
createTD()
print(len(training_data))
mid = (len(training_data)/2)

#take data out of training data and make test data
test= []
for i in range(50,-50,-1):
    test.append(training_data.pop(int(mid+i)))
print(len(training_data))
print(len(test))

#shuffle the list of data
random.shuffle(training_data)
random.shuffle(test)


x=[]
y=[]
for features,label in training_data: 
    x.append(features)#the actaul info in the data
    y.append(label) # 1(without mask) or 0 (with mask)

xTest = []
yTest = []
for feature,label in test:
    xTest.append(features)
    yTest.append(label)
    
    
#noramlize data between 0-1
x = norm(x)
xTest = norm(xTest)
input_shape = [SIZE,SIZE,1]


#// build model
print("building model")
model = keras.Sequential(); 
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
print("compiling model")

# compile model
opt = Adam(lr=.0001, decay=.0004 / 29)
model.compile(loss="mse",
             optimizer=opt,
             metrics=['accuracy'])
             
print("fitting model")       
#// run data
x = np.array(x).reshape(len(x),SIZE,SIZE,1).tolist()
H = model.fit(x,y,batch_size=128, epochs=5)
xTest = np.array(xTest).reshape(len(xTest),SIZE,SIZE,1).tolist()
print("evaluting")
loss, acc = model.evaluate(xTest,yTest)  # evaluate the out of sample data with model
print("Loss: ",loss)  # model's loss (error)
print("Accuracy: ", acc)  # model's accuracy





