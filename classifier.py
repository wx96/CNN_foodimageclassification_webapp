from PIL import Image
import numpy as np
import os
import cv2

# prepare dataset
data=[]
labels=[]
chickens=os.listdir("chickens")
for chicken in chickens:
    imag=cv2.imread("images/chickens/"+ chicken)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((200, 200))
    data.append(np.array(resized_image))
    labels.append(0)
kelis=os.listdir("kelis")
for keli in kelis:
    imag=cv2.imread("images/kelis/"+keli)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((200, 200))
    data.append(np.array(resized_image))
    labels.append(1)
rices=os.listdir("rices")
for rice in rices:
    imag=cv2.imread("images/rices/"+rice)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((200, 200))
    data.append(np.array(resized_image))
    labels.append(2)

# convert the normal aray of "data" and "labels" into numpy array
foods=np.array(data)
labels=np.array(labels)

# save the numpy array
np.save("foods",foods)
np.save("labels",labels)

# load the arrays
animals=np.load("foods.npy")
labels=np.load("labels.npy")

# shuffle "foods" and "labels" set
s=np.arange(foods.shape[0])
np.random.shuffle(s)
foods=foods[s]
labels=labels[s]

# total number of food categories
num_classes=len(np.unique(labels))
# size of dataset
data_length=len(foods)

# divide data into test and train
(x_train,x_test)=foods[(int)(0.1*data_length):],foods[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

# divide labelsinto test and train
(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]


import keras
from keras.utils import np_utils
#One hot encoding
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

# keras model
# import sequential model and all the required layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
#make model
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(200,200,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation="softmax"))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=50,epochs=100,verbose=1,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
model.save("model.h5")
print("Saved model to disk")

