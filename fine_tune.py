from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten

#dimensions of our image
img_width, image_height = 150, 150
top_model_weight_path = 'bottleneck_fc_model.h5'

# training and validation dataset target directories
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

# build a VGG16 network

model = applications.VGG16(include_top= False, weights='imagenet')
print('Model Loaded.')

# build a classifier model on top of convolutional model

top_model = Sequential()
# output of convolutional model bacomes the input of new classifier model
# created own classifier model with different features. include_top = False means classification layers of VGG16 model are not used.
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

top_model.load_weights(top_model_weight_path)

model.add(top_model)

# Freeze the first convolutional layers. Weights of convolutionals layers will not be updated.

for layers in model.layers[:25]:
    layers.trainable = False

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1. /255, shear_range=0.2, zoom_range=0.2, horizontal_flip= True)
validation_datagen = ImageDataGenerator(rescale=1. /255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(image_height, img_width), batch_size=batch_size, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(image_height, img_width), batch_size= batch_size, class_mode='binary')

model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs= epochs, validation_data=validation_generator,
                    validation_steps= nb_validation_samples // batch_size)

'''
Output:

D:\Anaconda3\envs\aind\python.exe C:/Users/Intruder/PycharmProjects/bottleneck_model/implementing_bottleneck.py
Using TensorFlow backend.
2017-09-02 21:56:07.020427: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 21:56:07.022685: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 21:56:07.023119: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 21:56:07.023535: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 21:56:07.023975: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 21:56:07.024411: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 21:56:07.024836: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 21:56:07.025242: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Found 2000 images belonging to 2 classes.
Found 802 images belonging to 2 classes.
Train data after loading is :  [[[[ 0.48350725  0.          0.         ...,  0.          0.82917339  0.        ]
   [ 0.03485662  0.          0.22486612 ...,  0.          1.03897095  0.        ]
   [ 0.31540334  0.          0.         ...,  0.          0.49017781  0.        ]
   [ 0.234595    0.          0.         ...,  0.          0.28118539  0.        ]]

  [[ 0.19103214  0.          0.         ...,  0.          0.829377    0.        ]
   [ 0.          0.          1.21792376 ...,  0.          1.10759044  0.        ]
   [ 0.03068137  0.          1.43744636 ...,  0.          0.34353119  0.        ]
   [ 0.57297218  0.          0.         ...,  0.          0.          0.        ]]

  [[ 0.1693598   0.          0.         ...,  0.          0.27577502  0.        ]
   [ 0.          0.          1.71624732 ...,  0.          0.31318465  0.        ]
   [ 0.1563338   0.          1.87400031 ...,  0.          0.41894716  0.        ]
   [ 0.65035462  0.          0.95875335 ...,  0.          0.01588911  0.        ]]

  [[ 0.44808903  0.          0.         ...,  0.          0.30772585  0.        ]
   [ 0.43148249  0.          1.05940938 ...,  0.          0.51654887  0.        ]
   [ 0.2360073   0.          1.3714776  ...,  0.          0.84349477  0.        ]
   [ 0.          0.          1.32231438 ...,  0.30869505  0.59621906  0.        ]]]


 [[[ 0.24343485  0.          0.         ...,  0.          0.71908289  0.        ]
   [ 0.59764481  0.          0.         ...,  0.          0.26454937  0.        ]
   [ 0.49105084  0.          0.         ...,  0.          0.14229232  0.        ]
   [ 0.67472309  0.          0.         ...,  0.          0.10808563  0.        ]]

  [[ 0.38266289  0.          0.15404546 ...,  0.          0.62203526  0.        ]
   [ 0.11067306  0.          0.59517932 ...,  0.          0.11322635  0.        ]
   [ 0.          0.          0.04054642 ...,  0.          0.46884492  0.        ]
   [ 0.61899841  0.          0.         ...,  0.          0.61292011  0.        ]]

  [[ 0.70318222  0.          0.07879525 ...,  0.          0.76679754  0.        ]
   [ 0.          0.          0.         ...,  0.          0.          0.        ]
   [ 0.          0.          0.         ...,  0.          0.3022466   0.        ]
   [ 0.34029299  0.          0.18551683 ...,  0.          0.79176182  0.        ]]

  [[ 0.5711695   0.          0.         ...,  0.          0.75989389  0.        ]
   [ 0.36337301  0.          0.         ...,  0.          0.40211314  0.        ]
   [ 0.44617844  0.          0.         ...,  0.          0.23802692  0.        ]
   [ 0.18213925  0.          0.13623402 ...,  0.          0.94349849  0.        ]]]


 [[[ 0.74064291  0.          0.04968229 ...,  0.          0.82942164  0.        ]
   [ 0.38173741  0.          0.         ...,  0.          0.9657222   0.        ]
   [ 0.10870712  0.          0.5125947  ...,  0.          0.8228091   0.        ]
   [ 0.          0.          0.27786934 ...,  0.          0.          0.        ]]

  [[ 0.68506879  0.          0.43191898 ...,  0.          0.28474313  0.        ]
   [ 0.02673262  0.          0.67453676 ...,  0.          0.21954662  0.        ]
   [ 0.          0.          1.16991866 ...,  0.          0.42357564  0.        ]
   [ 0.          0.          0.77510643 ...,  0.          0.          0.        ]]

  [[ 0.          0.          1.01037598 ...,  0.          0.60066861  0.        ]
   [ 0.          0.          0.78873372 ...,  0.          0.          0.        ]
   [ 0.36155242  0.          0.         ...,  0.27750936  0.00598782  0.        ]
   [ 0.          0.          0.03039622 ...,  0.0610236   0.44905946  0.        ]]

  [[ 0.          0.          0.86082387 ...,  0.          0.7149114   0.        ]
   [ 0.06878991  0.          0.76691681 ...,  0.          0.07629871  0.        ]
   [ 1.18790174  0.          0.41162884 ...,  0.          0.41223997  0.        ]
   [ 0.72828233  0.          0.47152045 ...,  0.          0.60659146  0.        ]]]


 ..., 
 [[[ 0.          0.          0.         ...,  0.          1.19647241  0.        ]
   [ 0.          0.          0.         ...,  0.          0.79178262  0.        ]
   [ 0.          0.          0.27549845 ...,  0.          0.6700471   0.        ]
   [ 0.17056073  0.          1.29371953 ...,  0.12949437  0.1947512   0.        ]]

  [[ 0.          0.          0.         ...,  0.          0.93174696  0.        ]
   [ 0.          0.          0.21461615 ...,  0.          0.26861182  0.        ]
   [ 0.          0.          0.83560264 ...,  0.04974432  0.74947041  0.        ]
   [ 0.20213015  0.          2.38842106 ...,  0.30305806  0.56423473  0.        ]]

  [[ 0.          0.          0.02561277 ...,  0.          0.18143505  0.        ]
   [ 0.          0.          0.17047787 ...,  0.          0.          0.        ]
   [ 0.30384344  0.          0.47123015 ...,  0.          0.36905274  0.        ]
   [ 0.48128957  0.          1.74645138 ...,  0.          0.72314805  0.        ]]

  [[ 0.          0.          0.         ...,  0.          0.51840192  0.        ]
   [ 0.04729725  0.          0.         ...,  0.          0.          0.        ]
   [ 1.4688803   0.          0.89571655 ...,  0.          0.          0.        ]
   [ 1.65710819  0.          0.52162099 ...,  0.          0.          0.        ]]]


 [[[ 0.15465294  0.          0.         ...,  0.          0.80982184  0.        ]
   [ 0.44510329  0.          0.42467368 ...,  0.0463535   0.53439605  0.        ]
   [ 0.43989095  0.          0.58015728 ...,  0.37784174  0.59448022  0.        ]
   [ 0.14241165  0.          0.29283118 ...,  0.10387842  0.69344902  0.        ]]

  [[ 0.28505623  0.          0.58329999 ...,  0.          0.86150014  0.        ]
   [ 0.49965906  0.          1.32830286 ...,  0.37824166  0.34228373  0.        ]
   [ 0.54553723  0.          1.49952531 ...,  0.80630875  0.17318484  0.        ]
   [ 0.43222001  0.          1.10225749 ...,  0.55502844  0.43046772  0.        ]]

  [[ 0.          0.          0.75501752 ...,  0.          0.72844493  0.        ]
   [ 0.          0.          1.42737007 ...,  0.31007251  0.21935457  0.        ]
   [ 0.          0.          1.3854332  ...,  0.08093959  0.16688418  0.        ]
   [ 0.29523206  0.          0.98221254 ...,  0.          0.17108494  0.        ]]

  [[ 0.          0.          0.24252293 ...,  0.          0.46314892  0.        ]
   [ 0.          0.          0.76754051 ...,  0.20140672  0.          0.        ]
   [ 0.40693957  0.          0.82773864 ...,  0.          0.50089806  0.        ]
   [ 0.5171299   0.          0.47879109 ...,  0.          0.52640939  0.        ]]]


 [[[ 0.33781648  0.          0.         ...,  0.          0.72190571  0.        ]
   [ 0.66936588  0.          0.63851172 ...,  0.62694895  0.46536803  0.        ]
   [ 0.21631269  0.          1.54914951 ...,  0.51280773  0.70581752  0.        ]
   [ 0.          0.          1.31538796 ...,  0.          0.55799621  0.        ]]

  [[ 0.          0.          0.00883913 ...,  0.40895757  0.44412684  0.        ]
   [ 1.25321054  0.          0.53495556 ...,  1.26296222  0.          0.        ]
   [ 1.77862501  0.          1.45037317 ...,  1.03952527  0.57764232  0.        ]
   [ 0.98860729  0.          1.46426713 ...,  0.03808051  0.4825947
     0.10377243]]

  [[ 0.89918029  0.          0.21461597 ...,  0.          0.          0.        ]
   [ 1.57153308  0.          0.         ...,  0.          0.          0.        ]
   [ 1.8199116   0.          0.55331445 ...,  0.          0.78412127  0.        ]
   [ 0.99303889  0.          0.70459223 ...,  0.          0.72495371  0.        ]]

  [[ 1.3933742   0.          0.21758774 ...,  0.          0.69323623  0.        ]
   [ 1.69056857  0.          0.35357752 ...,  0.          0.          0.        ]
   [ 1.48501182  0.          0.35679144 ...,  0.          0.44103861  0.        ]
   [ 1.51153076  0.          0.36551011 ...,  0.          0.33810174  0.        ]]]]
Shape of trained data is :  (4, 4, 512)
Training labeld are : [0 0 0 ..., 1 1 1]
Train on 2000 samples, validate on 800 samples
Epoch 1/50
5s - loss: 0.7461 - acc: 0.7540 - val_loss: 0.3397 - val_acc: 0.8375
Epoch 2/50
4s - loss: 0.3682 - acc: 0.8510 - val_loss: 0.2565 - val_acc: 0.9062
Epoch 3/50
4s - loss: 0.3033 - acc: 0.8805 - val_loss: 0.4757 - val_acc: 0.8037
Epoch 4/50
4s - loss: 0.2613 - acc: 0.8990 - val_loss: 0.2521 - val_acc: 0.9062
Epoch 5/50
4s - loss: 0.2264 - acc: 0.9145 - val_loss: 0.2470 - val_acc: 0.9087
Epoch 6/50
4s - loss: 0.1803 - acc: 0.9310 - val_loss: 0.3103 - val_acc: 0.9025
Epoch 7/50
4s - loss: 0.1695 - acc: 0.9365 - val_loss: 0.2856 - val_acc: 0.9038
Epoch 8/50
4s - loss: 0.1543 - acc: 0.9400 - val_loss: 0.3558 - val_acc: 0.8862
Epoch 9/50
4s - loss: 0.1409 - acc: 0.9470 - val_loss: 0.4142 - val_acc: 0.8838
Epoch 10/50
4s - loss: 0.1274 - acc: 0.9445 - val_loss: 0.4498 - val_acc: 0.8775
Epoch 11/50
4s - loss: 0.1050 - acc: 0.9550 - val_loss: 0.7828 - val_acc: 0.8163
Epoch 12/50
4s - loss: 0.0879 - acc: 0.9695 - val_loss: 0.4420 - val_acc: 0.9000
Epoch 13/50
4s - loss: 0.0973 - acc: 0.9665 - val_loss: 0.5563 - val_acc: 0.8800
Epoch 14/50
5s - loss: 0.0832 - acc: 0.9675 - val_loss: 0.6082 - val_acc: 0.8712
Epoch 15/50
5s - loss: 0.0720 - acc: 0.9730 - val_loss: 0.5162 - val_acc: 0.8988
Epoch 16/50
4s - loss: 0.0634 - acc: 0.9770 - val_loss: 0.4783 - val_acc: 0.8975
Epoch 17/50
4s - loss: 0.0479 - acc: 0.9785 - val_loss: 0.5225 - val_acc: 0.8962
Epoch 18/50
4s - loss: 0.0544 - acc: 0.9835 - val_loss: 0.5764 - val_acc: 0.8975
Epoch 19/50
4s - loss: 0.0584 - acc: 0.9760 - val_loss: 0.5487 - val_acc: 0.8988
Epoch 20/50
4s - loss: 0.0601 - acc: 0.9805 - val_loss: 0.5457 - val_acc: 0.8988
Epoch 21/50
4s - loss: 0.0344 - acc: 0.9865 - val_loss: 0.6916 - val_acc: 0.8950
Epoch 22/50
4s - loss: 0.0476 - acc: 0.9840 - val_loss: 0.7278 - val_acc: 0.8962
Epoch 23/50
4s - loss: 0.0330 - acc: 0.9920 - val_loss: 0.8223 - val_acc: 0.8725
Epoch 24/50
5s - loss: 0.0260 - acc: 0.9910 - val_loss: 1.2004 - val_acc: 0.8500
Epoch 25/50
5s - loss: 0.0505 - acc: 0.9840 - val_loss: 0.6930 - val_acc: 0.8988
Epoch 26/50
4s - loss: 0.0365 - acc: 0.9885 - val_loss: 0.7014 - val_acc: 0.9012
Epoch 27/50
4s - loss: 0.0290 - acc: 0.9880 - val_loss: 0.7215 - val_acc: 0.8850
Epoch 28/50
4s - loss: 0.0260 - acc: 0.9905 - val_loss: 0.7264 - val_acc: 0.9025
Epoch 29/50
4s - loss: 0.0184 - acc: 0.9940 - val_loss: 0.7198 - val_acc: 0.8988
Epoch 30/50
4s - loss: 0.0102 - acc: 0.9945 - val_loss: 0.7558 - val_acc: 0.8950
Epoch 31/50
4s - loss: 0.0355 - acc: 0.9920 - val_loss: 0.7484 - val_acc: 0.9062
Epoch 32/50
5s - loss: 0.0355 - acc: 0.9910 - val_loss: 0.7535 - val_acc: 0.9038
Epoch 33/50
5s - loss: 0.0136 - acc: 0.9935 - val_loss: 0.8405 - val_acc: 0.9012
Epoch 34/50
5s - loss: 0.0248 - acc: 0.9925 - val_loss: 0.8835 - val_acc: 0.8950
Epoch 35/50
4s - loss: 0.0306 - acc: 0.9925 - val_loss: 0.8454 - val_acc: 0.8925
Epoch 36/50
4s - loss: 0.0268 - acc: 0.9925 - val_loss: 1.2634 - val_acc: 0.8475
Epoch 37/50
4s - loss: 0.0124 - acc: 0.9970 - val_loss: 0.9249 - val_acc: 0.8975
Epoch 38/50
4s - loss: 0.0180 - acc: 0.9945 - val_loss: 0.9593 - val_acc: 0.8875
Epoch 39/50
5s - loss: 0.0164 - acc: 0.9950 - val_loss: 0.8665 - val_acc: 0.9012
Epoch 40/50
5s - loss: 0.0191 - acc: 0.9955 - val_loss: 0.8349 - val_acc: 0.9000
Epoch 41/50
5s - loss: 0.0078 - acc: 0.9955 - val_loss: 0.9449 - val_acc: 0.8938
Epoch 42/50
4s - loss: 0.0239 - acc: 0.9940 - val_loss: 0.9235 - val_acc: 0.8912
Epoch 43/50
5s - loss: 0.0082 - acc: 0.9980 - val_loss: 0.8814 - val_acc: 0.9075
Epoch 44/50
4s - loss: 0.0123 - acc: 0.9960 - val_loss: 0.9133 - val_acc: 0.9075
Epoch 45/50
4s - loss: 0.0078 - acc: 0.9980 - val_loss: 0.9868 - val_acc: 0.8925
Epoch 46/50
4s - loss: 0.0109 - acc: 0.9970 - val_loss: 0.8667 - val_acc: 0.9038
Epoch 47/50
4s - loss: 0.0155 - acc: 0.9960 - val_loss: 0.8966 - val_acc: 0.9050
Epoch 48/50
4s - loss: 0.0089 - acc: 0.9975 - val_loss: 0.9998 - val_acc: 0.8888
Epoch 49/50
4s - loss: 0.0155 - acc: 0.9975 - val_loss: 1.0416 - val_acc: 0.8888
Epoch 50/50
5s - loss: 0.0165 - acc: 0.9970 - val_loss: 1.0296 - val_acc: 0.8938

Process finished with exit code 0

'''