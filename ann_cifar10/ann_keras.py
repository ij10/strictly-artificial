# install keras with: pip3 install keras


# import packages

import keras
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import matplotlib.pyplot as plt

# load the CIFAR-10 data

(x_train, y_label_train), (x_test, y_label_test) = cifar10.load_data()


# transform the labels to one-hot labels

y_train = to_categorical(y_label_train, 10)
y_test = to_categorical(y_label_test, 10)


# scale the input data between 0 and 1

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# create a model function

model = Sequential()

model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


# train the model

history = model.fit(x_train, y_train, epochs=50, batch_size=100, validation_split=0.1) # , validation_data=(x_test, y_test))


# save the model

model.save("my_model.h5")


# evaluate the model

score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])


# print the keys stored in the fitted model

print(history.history.keys())


# plot the accuracy and loss

# accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('epoch vs accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'])
plt.show()

# loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('epoch vs loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'])
plt.show()
