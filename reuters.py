#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 03:44:42 2019

@author: berk
"""

from keras import datasets

(train_data , train_labels),(test_data,test_labels)= datasets.reuters.load_data(num_words=10000)

#vectorize data

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences),dimension))
        for i, sequence in enumerate(sequences):
                results[i,sequence]=1.

        return results

x_train= vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#vectorize labels
def to_one_hot (labels,dimensions=46):
        results = np.zeros((len(labels),dimensions))
        for i, label in enumerate(labels):
                results[i,label]=1.
        return results

y_train = to_one_hot(train_labels)
y_test  = to_one_hot(test_labels)
#from keras.utils.np_utils import to_categorical
#y_train = to_categorical(train_labels)
#y_test = to_categorical(test_labels)


#build model

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(46,activation="softmax"))


model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])


#initial validating dataset

x_val = x_train[:1000] #1000 sample from the end
partial_x_train = x_train[1000:] #rest of the data

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model.fit(partial_x_train,partial_y_train,epochs=9,batch_size=512,validation_data=(x_val,y_val))

predictions = model.predict(x_test)
np.argmax(predictions[0])+1  #class of first test element


# visualize the loss

import matplotlib.pyplot as plt

loss = model.history.history["loss"]
val_loss = model.history.history["val_loss"]
epochs = range(1, len(loss)+1)
plt.plot(epochs , loss ,"bo",label="Train loss")
plt.plot(epochs , val_loss,"b",label="Validate loss")
plt.xlabel("epochs")
plt.ylabel("losses")
plt.legend()
plt.show()


# visualize the acc

plt.clf()
acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']
plt.plot(epochs,acc,"bo",label="Acc")
plt.plot(epochs,val_acc,"b",label="validate acc")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracies")
plt.show()
































