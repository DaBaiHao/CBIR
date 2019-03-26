
# Imports
import numpy as np
import keras
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D


batch_size = 32
num_classes = 10
data_augmentation = True

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create the model now
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(48, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(48, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(60, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(60, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(60, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(60, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(60, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(80, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(80, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(80, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(80, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(80, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(128, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(128, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(128, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(Conv2D(128, (3, 3), padding='same',
                                 activation='relu',input_shape=x_train.shape[1:]))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))

model.add(Dense(500, activation='relu',))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax',))
model.summary ()

# initiate RMSprop optimizer
opt = keras.optimizers.Adam(lr=0.0001)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])

print("train____________")
model.fit(x_train,y_train,epochs=90,batch_size=128,)
print("test_____________")
model.save('my_model.h5')
loss,acc=model.evaluate(x_test,y_test)
print("loss=",loss)
print("accuracy=",acc)