import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

"""
Using bottleneck features of a pre-trained network : 90% accuracy in a minute

A most refined approach would be leverage a model pre-trained on a larger dataset. Such a network would have already learned features that are useful for most 
computer vision problems, and leveraging such features would allow us to reach a better accuracy than any method that would only rely on the available data.

We will use the VGG16 architecture, pre-trained on the ImageNet dataset --a model previously featured on this blog. 
Because the ImageNet dataset contains several "cat" classes (persian cat, siamese cat...) and many "dog" classes among its total of 1000 classes, 
this model will already have learned features that are relevant to our classification problem. In fact, it is possible that merely recording the softmax 
predictions of the model over our data rather than the bottleneck features would be enough to solve our dogs vs. cats classification problem extremely well. 
However, the method we present here is more likely to generalize well to a broader range of problems, including problems featuring classes absent from ImageNet.

Our strategy will be as follow: we will only instantiate the convolutional part of the model, everything up to the fully-connected layers. 
We will then run this model on our training and validation data once, recording the output (the "bottleneck features" from th VGG16 model: 
the last activation maps before the fully-connected layers) in two numpy arrays. Then we will train a small fully-connected model on top of the stored features.


The reason why we are storing the features offline rather than adding our fully-connected model directly on top of a frozen convolutional base and running
 the whole thing, is computational effiency. Running VGG16 is expensive, especially if you're working on CPU, and we want to only do it once. 
Note that this prevents us from using data augmentation.

"""

# dimensions for our image
image_width, image_height = 150, 150

top_model_weight_paths = 'bottleneck_fc_model.h5'

train_data_dir = 'data/train' #directory of training data
validation_data_dir = 'data/validation' #directory of validation data
nb_train_samples = 2000 # no. of training samples
nb_validation_samples = 800 # no. of validation samples
epochs = 50 #no. of epochs
batch_size = 16 # batch size count
# Created a seprate method to save the bottleneck features
def save_bottleneck_features():
    # Create object of ImageDataGenerator class with only one feature i.e. rescaling
    datagen = ImageDataGenerator(rescale=1. /255)
    # as per our statergy we will initiate only convolutional part of model
    # Calling VGG16 method instantiates the VGG16 architecture
    # include_top : whether to connect 3 fully connected layer on the top of model or not. if include_top is False, the fully connected Dense layer part will
    # not be excuted.
    # Weights: imagenet(pre-training on imagenet)
    model = applications.VGG16(include_top=False, weights='imagenet')
    # loading images from directory and creating a training generator object
    train_generator = datagen.flow_from_directory(train_data_dir, target_size=(image_width, image_height),
                                            batch_size= batch_size,class_mode=None
                                            , shuffle=False) # Shuffle has been kept as false because we want dataset to in sequence so that labels can be determined effectively
    # predict_generator : generates predictions for input samples from data generator
    # dataset is ran on the already existing VGG16 model and predictions are generated
    bottleneck_feature_train = model.predict_generator(train_generator, nb_train_samples // batch_size)
    # Features are stored offline in numpy array
    np.save(open('bottleneck_feature_train.npy', 'wb'), bottleneck_feature_train)
    # Similarly validation generator is created
    validation_generator = datagen.flow_from_directory(validation_data_dir,
                                                       target_size=(image_width, image_height),
                                                       batch_size = batch_size, class_mode=None,
                                                       shuffle=False)
    bottleneck_feature_validation = model.predict_generator(validation_generator, nb_validation_samples // batch_size)
    # Validation features are stored offline in another numpy array
    np.save(open('bottleneck_feature_validation.npy', 'wb'), bottleneck_feature_validation)


def train_top_models():
    # Features stored offline are read from numpy array which was stored offline
    train_data = np.load(open('bottleneck_feature_train.npy', 'rb'))
    print("Train data after loading is : ", train_data)
    print("Shape of trained data is : ",train_data.shape[1:])
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
    print("Training labeld are :",train_labels)
    validation_data = np.load(open('bottleneck_feature_validation.npy', 'rb'))
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))
    model = Sequential()

    # Important thing to learn in this network is that we have removed the last fully connected layer of VGG16 network
    # and used our own dense layer. Where the learned data set saved in a file previously act as a input to this layer.
    # We will use the predictions learned in convolutional layers in the above method (We can claim convolutional layers are only used because value of no_top was true)
    # Learned generator is saved in the bottleneck_feature_train.npy and bottleneck_feature_validation.npy
    # We will read this data from the corrosponding files and this will become the input to the final dense layer
    # That dense layer we wil define ourself. Ideally its the output layer.
    # this layer has few other catagories with one of the each label we wanted to detect.
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
              validation_data=(validation_data, validation_labels), verbose=2)
    # We train only the weights in this layer i.e. connected layer freezing the weights in previous convolutional layers.
    model.save_weights(top_model_weight_paths)

save_bottleneck_features()
train_top_models()
"""
Output:

D:\Anaconda3\envs\aind\python.exe C:/Users/Intruder/PycharmProjects/bottleneck_model/implementing_bottleneck.py
Using TensorFlow backend.
2017-09-02 16:01:27.004518: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 16:01:27.004799: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 16:01:27.005055: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 16:01:27.005310: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 16:01:27.005587: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 16:01:27.005848: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 16:01:27.006105: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-09-02 16:01:27.006363: W c:\l\tensorflow_1501918863922\work\tensorflow-1.2.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
Found 2000 images belonging to 2 classes.
Found 802 images belonging to 2 classes.
Train data after loading is :  
    [[[[ 0.48350725  0.          0.         ...,  0.          0.82917339  0.        ]
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
5s - loss: 0.5461 - acc: 0.7715 - val_loss: 0.9948 - val_acc: 0.5875
Epoch 2/50
4s - loss: 0.3539 - acc: 0.8530 - val_loss: 0.2812 - val_acc: 0.8875
Epoch 3/50
5s - loss: 0.2889 - acc: 0.8935 - val_loss: 0.2531 - val_acc: 0.9100
Epoch 4/50
4s - loss: 0.2391 - acc: 0.9075 - val_loss: 0.2757 - val_acc: 0.9050
Epoch 5/50
4s - loss: 0.2314 - acc: 0.9140 - val_loss: 0.3367 - val_acc: 0.8800
Epoch 6/50
4s - loss: 0.2088 - acc: 0.9265 - val_loss: 0.2967 - val_acc: 0.9000
Epoch 7/50
4s - loss: 0.1559 - acc: 0.9425 - val_loss: 0.4380 - val_acc: 0.8762
Epoch 8/50
4s - loss: 0.1584 - acc: 0.9415 - val_loss: 0.4253 - val_acc: 0.8825
Epoch 9/50
4s - loss: 0.1198 - acc: 0.9575 - val_loss: 0.3980 - val_acc: 0.8962
Epoch 10/50
4s - loss: 0.1320 - acc: 0.9565 - val_loss: 0.4675 - val_acc: 0.8912
Epoch 11/50
4s - loss: 0.1102 - acc: 0.9625 - val_loss: 0.5254 - val_acc: 0.8750
Epoch 12/50
4s - loss: 0.0984 - acc: 0.9660 - val_loss: 0.8964 - val_acc: 0.8325
Epoch 13/50
5s - loss: 0.0770 - acc: 0.9685 - val_loss: 0.5450 - val_acc: 0.8862
Epoch 14/50
5s - loss: 0.0790 - acc: 0.9760 - val_loss: 0.7661 - val_acc: 0.8625
Epoch 15/50
4s - loss: 0.0793 - acc: 0.9775 - val_loss: 0.5813 - val_acc: 0.8875
Epoch 16/50
4s - loss: 0.0626 - acc: 0.9770 - val_loss: 0.6307 - val_acc: 0.9012
Epoch 17/50
4s - loss: 0.0529 - acc: 0.9775 - val_loss: 0.7082 - val_acc: 0.8762
Epoch 18/50
4s - loss: 0.0456 - acc: 0.9845 - val_loss: 1.0189 - val_acc: 0.8538
Epoch 19/50
4s - loss: 0.0560 - acc: 0.9850 - val_loss: 0.6608 - val_acc: 0.8912
Epoch 20/50
4s - loss: 0.0475 - acc: 0.9825 - val_loss: 0.5808 - val_acc: 0.8988
Epoch 21/50
4s - loss: 0.0378 - acc: 0.9835 - val_loss: 0.6082 - val_acc: 0.8875
Epoch 22/50
4s - loss: 0.0288 - acc: 0.9905 - val_loss: 0.7644 - val_acc: 0.8838
Epoch 23/50
4s - loss: 0.0257 - acc: 0.9900 - val_loss: 0.8027 - val_acc: 0.8912
Epoch 24/50
4s - loss: 0.0312 - acc: 0.9900 - val_loss: 0.7524 - val_acc: 0.8975
Epoch 25/50
4s - loss: 0.0377 - acc: 0.9870 - val_loss: 0.7131 - val_acc: 0.8900
Epoch 26/50
4s - loss: 0.0257 - acc: 0.9910 - val_loss: 0.7228 - val_acc: 0.8950
Epoch 27/50
4s - loss: 0.0391 - acc: 0.9875 - val_loss: 0.7755 - val_acc: 0.8900
Epoch 28/50
4s - loss: 0.0242 - acc: 0.9930 - val_loss: 0.8920 - val_acc: 0.8800
Epoch 29/50
4s - loss: 0.0228 - acc: 0.9910 - val_loss: 0.8689 - val_acc: 0.8875
Epoch 30/50
4s - loss: 0.0290 - acc: 0.9905 - val_loss: 0.8084 - val_acc: 0.8875
Epoch 31/50
4s - loss: 0.0258 - acc: 0.9935 - val_loss: 0.8035 - val_acc: 0.8938
Epoch 32/50
4s - loss: 0.0247 - acc: 0.9925 - val_loss: 0.8156 - val_acc: 0.8938
Epoch 33/50
4s - loss: 0.0201 - acc: 0.9940 - val_loss: 0.8351 - val_acc: 0.8925
Epoch 34/50
4s - loss: 0.0132 - acc: 0.9945 - val_loss: 1.2554 - val_acc: 0.8550
Epoch 35/50
4s - loss: 0.0251 - acc: 0.9920 - val_loss: 0.8227 - val_acc: 0.8962
Epoch 36/50
4s - loss: 0.0241 - acc: 0.9950 - val_loss: 0.9139 - val_acc: 0.8888
Epoch 37/50
4s - loss: 0.0134 - acc: 0.9950 - val_loss: 1.0158 - val_acc: 0.8788
Epoch 38/50
4s - loss: 0.0182 - acc: 0.9950 - val_loss: 0.9354 - val_acc: 0.8912
Epoch 39/50
4s - loss: 0.0185 - acc: 0.9945 - val_loss: 0.8365 - val_acc: 0.8988
Epoch 40/50
4s - loss: 0.0075 - acc: 0.9960 - val_loss: 0.9551 - val_acc: 0.8962
Epoch 41/50
4s - loss: 0.0317 - acc: 0.9910 - val_loss: 0.8488 - val_acc: 0.8938
Epoch 42/50
4s - loss: 0.0108 - acc: 0.9975 - val_loss: 0.8859 - val_acc: 0.8912
Epoch 43/50
4s - loss: 0.0197 - acc: 0.9940 - val_loss: 0.8575 - val_acc: 0.8938
Epoch 44/50
4s - loss: 0.0173 - acc: 0.9945 - val_loss: 0.9127 - val_acc: 0.8925
Epoch 45/50
4s - loss: 0.0180 - acc: 0.9940 - val_loss: 0.9872 - val_acc: 0.8888
Epoch 46/50
4s - loss: 0.0118 - acc: 0.9960 - val_loss: 1.0706 - val_acc: 0.8888
Epoch 47/50
4s - loss: 0.0176 - acc: 0.9960 - val_loss: 0.9963 - val_acc: 0.8938
Epoch 48/50
4s - loss: 0.0115 - acc: 0.9945 - val_loss: 0.9579 - val_acc: 0.8900
Epoch 49/50
4s - loss: 0.0163 - acc: 0.9975 - val_loss: 1.1297 - val_acc: 0.8850
Epoch 50/50
4s - loss: 0.0096 - acc: 0.9965 - val_loss: 1.0159 - val_acc: 0.8900

Process finished with exit code 0

"""