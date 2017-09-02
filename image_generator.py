from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as k

# dimensions for our image
# these are the dimensions of the image. We know that real world images can be of any dimensions. We have taken width and height of image as standand dimensions
image_width, image_height = 150, 150

# Data has to be divided in train and validation
train_data_directory = 'data/train' # Raw images has been divided to train and validation. Train directory
validation_data_directory = 'data/validation' # validation data has been generated from training data set and placed under different directory

nb_train_samples = 2000 # number of image samples for training
nb_validation_samples = 800 # number of image samples for validation
# there is the huge right up abt epoch in the below link. Number of epoch is decided on the base where training loss sets to decrease but validation loss still increases
epochs = 50 # https://github.com/nishantmittal1990/CNN_Pods/blob/master/mnist-mlp/mnist_mlp.ipynb
# we have defined epochs initially as 50 but these keep on changing
batch_size = 16 #number of images we want in batch

if k.image_data_format() == 'channels_first':
    input_shape = (3, image_width, image_height)
else:
    input_shape = (image_width, image_height, 3)


model = Sequential()
# Number of filters = 32
# kernel_size = 2,2
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 16
# argument configuration we used for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# argument configuration we used for testing only the rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
# images will flow directly from target directory
# images are resized to 150*150
# batches for argumented images
# since we are using binary cross entropy. we need binary labels
train_generator = train_datagen.flow_from_directory(train_data_directory, target_size=(image_width, image_height),batch_size=batch_size, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_data_directory, target_size=(image_width, image_height), batch_size=batch_size, class_mode='binary')
# steps_per_epoch - Number of unique samples in dataset divided by batch size
# We do this to make sure model see x_train.shape[0] argumented images in each epoch
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='first_try.h5',verbose=1, save_best_only=True)
model.fit_generator(train_generator, steps_per_epoch= nb_train_samples // batch_size, epochs=epochs, verbose=2,
                    callbacks=[checkpointer],
                    validation_data=validation_generator, validation_steps= nb_validation_samples // batch_size)
#model.save_weights('first_try.h5')


"""
Output:
D:\Anaconda3\envs\aind\python.exe C:/Users/Intruder/PycharmProjects/bottleneck_model/image_generator.py
Using TensorFlow backend.
Found 2000 images belonging to 2 classes.
Found 802 images belonging to 2 classes.
Epoch 1/50
2017-09-01 21:47:44.156654: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
2017-09-01 21:47:44.158013: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-01 21:47:44.159323: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-01 21:47:44.160533: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-01 21:47:44.161840: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-01 21:47:44.163151: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-01 21:47:44.164457: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-01 21:47:44.165727: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Epoch 00000: val_loss improved from inf to 0.66419, saving model to first_try.h5
109s - loss: 0.7463 - acc: 0.5400 - val_loss: 0.6642 - val_acc: 0.6362
Epoch 2/50
Epoch 00001: val_loss did not improve
101s - loss: 0.6643 - acc: 0.6210 - val_loss: 0.6881 - val_acc: 0.5483
Epoch 3/50
Epoch 00002: val_loss did not improve
99s - loss: 0.6307 - acc: 0.6570 - val_loss: 0.6762 - val_acc: 0.5891
Epoch 4/50
Epoch 00003: val_loss improved from 0.66419 to 0.61506, saving model to first_try.h5
104s - loss: 0.6096 - acc: 0.6970 - val_loss: 0.6151 - val_acc: 0.6374
Epoch 5/50
Epoch 00004: val_loss did not improve
106s - loss: 0.5787 - acc: 0.6995 - val_loss: 0.6242 - val_acc: 0.6463
Epoch 6/50
Epoch 00005: val_loss improved from 0.61506 to 0.53046, saving model to first_try.h5
104s - loss: 0.5597 - acc: 0.7240 - val_loss: 0.5305 - val_acc: 0.7087
Epoch 7/50
Epoch 00006: val_loss did not improve
107s - loss: 0.5566 - acc: 0.7270 - val_loss: 0.5767 - val_acc: 0.7087
Epoch 8/50
Epoch 00007: val_loss improved from 0.53046 to 0.52459, saving model to first_try.h5
111s - loss: 0.5406 - acc: 0.7325 - val_loss: 0.5246 - val_acc: 0.7430
Epoch 9/50
Epoch 00008: val_loss did not improve
105s - loss: 0.5301 - acc: 0.7435 - val_loss: 0.5313 - val_acc: 0.7226
Epoch 10/50
Epoch 00009: val_loss improved from 0.52459 to 0.49935, saving model to first_try.h5
105s - loss: 0.5165 - acc: 0.7585 - val_loss: 0.4994 - val_acc: 0.7646
Epoch 11/50
Epoch 00010: val_loss did not improve
103s - loss: 0.5012 - acc: 0.7595 - val_loss: 0.5343 - val_acc: 0.7252
Epoch 12/50
Epoch 00011: val_loss did not improve
103s - loss: 0.5037 - acc: 0.7650 - val_loss: 0.5861 - val_acc: 0.7430
Epoch 13/50
Epoch 00012: val_loss did not improve
106s - loss: 0.4978 - acc: 0.7625 - val_loss: 0.5781 - val_acc: 0.7316
Epoch 14/50
Epoch 00013: val_loss did not improve
108s - loss: 0.4797 - acc: 0.7750 - val_loss: 0.5133 - val_acc: 0.7774
Epoch 15/50
Epoch 00014: val_loss did not improve
109s - loss: 0.4780 - acc: 0.7810 - val_loss: 0.4998 - val_acc: 0.7837
Epoch 16/50
Epoch 00015: val_loss did not improve
103s - loss: 0.4637 - acc: 0.7865 - val_loss: 0.6439 - val_acc: 0.7188
Epoch 17/50
Epoch 00016: val_loss did not improve
101s - loss: 0.4412 - acc: 0.8030 - val_loss: 0.5848 - val_acc: 0.7430
Epoch 18/50
Epoch 00017: val_loss did not improve
103s - loss: 0.4690 - acc: 0.7880 - val_loss: 0.5315 - val_acc: 0.7723
Epoch 19/50
Epoch 00018: val_loss improved from 0.49935 to 0.44163, saving model to first_try.h5
105s - loss: 0.4477 - acc: 0.8010 - val_loss: 0.4416 - val_acc: 0.7990
Epoch 20/50
Epoch 00019: val_loss did not improve
109s - loss: 0.4469 - acc: 0.7985 - val_loss: 0.4803 - val_acc: 0.7761
Epoch 21/50
Epoch 00020: val_loss did not improve
120s - loss: 0.4366 - acc: 0.7955 - val_loss: 0.5003 - val_acc: 0.7774
Epoch 22/50
Epoch 00021: val_loss did not improve
125s - loss: 0.4292 - acc: 0.8165 - val_loss: 0.6613 - val_acc: 0.7659
Epoch 23/50
Epoch 00022: val_loss did not improve
142s - loss: 0.4502 - acc: 0.8070 - val_loss: 0.4686 - val_acc: 0.7926
Epoch 24/50
Epoch 00023: val_loss did not improve
124s - loss: 0.4304 - acc: 0.8120 - val_loss: 0.4489 - val_acc: 0.7952
Epoch 25/50
Epoch 00024: val_loss did not improve
129s - loss: 0.4346 - acc: 0.8145 - val_loss: 0.6901 - val_acc: 0.7341
Epoch 26/50
Epoch 00025: val_loss did not improve
127s - loss: 0.4296 - acc: 0.8055 - val_loss: 0.4967 - val_acc: 0.7863
Epoch 27/50
Epoch 00026: val_loss did not improve
136s - loss: 0.4335 - acc: 0.8100 - val_loss: 0.4817 - val_acc: 0.8066
Epoch 28/50
Epoch 00027: val_loss did not improve
152s - loss: 0.3953 - acc: 0.8285 - val_loss: 0.5198 - val_acc: 0.7901
Epoch 29/50
Epoch 00028: val_loss did not improve
128s - loss: 0.4265 - acc: 0.8170 - val_loss: 0.4878 - val_acc: 0.7723
Epoch 30/50
Epoch 00029: val_loss did not improve
130s - loss: 0.4177 - acc: 0.8215 - val_loss: 0.5498 - val_acc: 0.7239
Epoch 31/50
Epoch 00030: val_loss did not improve
130s - loss: 0.4093 - acc: 0.8225 - val_loss: 0.7370 - val_acc: 0.7532
Epoch 32/50
Epoch 00031: val_loss did not improve
127s - loss: 0.4286 - acc: 0.8115 - val_loss: 0.5026 - val_acc: 0.8117
Epoch 33/50
Epoch 00032: val_loss did not improve
128s - loss: 0.4182 - acc: 0.8290 - val_loss: 0.5312 - val_acc: 0.7672
Epoch 34/50
Epoch 00033: val_loss did not improve
133s - loss: 0.4183 - acc: 0.8235 - val_loss: 1.0334 - val_acc: 0.7150
Epoch 35/50
Epoch 00034: val_loss did not improve
126s - loss: 0.4057 - acc: 0.8265 - val_loss: 0.4780 - val_acc: 0.8193
Epoch 36/50
Epoch 00035: val_loss did not improve
126s - loss: 0.3861 - acc: 0.8300 - val_loss: 0.5532 - val_acc: 0.7748
Epoch 37/50
Epoch 00036: val_loss did not improve
134s - loss: 0.4057 - acc: 0.8350 - val_loss: 0.7628 - val_acc: 0.7290
Epoch 38/50
Epoch 00037: val_loss did not improve
133s - loss: 0.4185 - acc: 0.8215 - val_loss: 0.4485 - val_acc: 0.7837
Epoch 39/50
Epoch 00038: val_loss did not improve
146s - loss: 0.4222 - acc: 0.8215 - val_loss: 0.5549 - val_acc: 0.7506
Epoch 40/50
Epoch 00039: val_loss did not improve
146s - loss: 0.4274 - acc: 0.8195 - val_loss: 0.5160 - val_acc: 0.7977
Epoch 41/50
Epoch 00040: val_loss did not improve
129s - loss: 0.4078 - acc: 0.8345 - val_loss: 0.5210 - val_acc: 0.7621
Epoch 42/50
Epoch 00041: val_loss did not improve
109s - loss: 0.4038 - acc: 0.8365 - val_loss: 0.5151 - val_acc: 0.7608
Epoch 43/50
Epoch 00042: val_loss did not improve
106s - loss: 0.4108 - acc: 0.8275 - val_loss: 0.4925 - val_acc: 0.7774
Epoch 44/50
Epoch 00043: val_loss did not improve
106s - loss: 0.4112 - acc: 0.8300 - val_loss: 0.5270 - val_acc: 0.7405
Epoch 45/50
Epoch 00044: val_loss did not improve
102s - loss: 0.3918 - acc: 0.8385 - val_loss: 0.5116 - val_acc: 0.7430
Epoch 46/50
Epoch 00045: val_loss did not improve
107s - loss: 0.3931 - acc: 0.8400 - val_loss: 0.5436 - val_acc: 0.7824
Epoch 47/50
Epoch 00046: val_loss did not improve
107s - loss: 0.3948 - acc: 0.8385 - val_loss: 0.5368 - val_acc: 0.7748
Epoch 48/50
Epoch 00047: val_loss did not improve
104s - loss: 0.3927 - acc: 0.8320 - val_loss: 0.6103 - val_acc: 0.7341
Epoch 49/50
Epoch 00048: val_loss did not improve
101s - loss: 0.3898 - acc: 0.8335 - val_loss: 0.5596 - val_acc: 0.7913
Epoch 50/50
Epoch 00049: val_loss did not improve
101s - loss: 0.3875 - acc: 0.8440 - val_loss: 0.4825 - val_acc: 0.7964

Process finished with exit code 0



"""