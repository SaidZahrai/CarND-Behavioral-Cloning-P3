import glob
import os
import csv
import cv2
import sys, getopt
import datetime

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, save_model, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, Activation, Cropping2D, BatchNormalization
from keras.utils import plot_model
from keras import regularizers
import keras.callbacks as callbacks


#%matplotlib inline

def get_checkpoint_path(training_dir):
    DT = datetime.datetime.now()
    dname = training_dir + "/checkpoints-{0}-{1}-{2}-{3}-{4}".format(DT.month,DT.day,DT.hour,DT.min,DT.second)
    os.mkdir(dname)
    return dname+"/model.ckpt"

def get_data_dirs(dir_path):
    dirs = glob.glob(dir_path)
    inputDirs = []
    for fname in dirs:
        if (os.path.exists(fname+"/driving_log.csv")):
            inputDirs.append(fname)
    return inputDirs

def read_cvs(dir_path):
    lines =[]
    with open(dir_path+"/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        line = reader.__next__()
        for line in reader:
            lines.append({'center': dir_path + '/IMG/' + line[0].split('/')[-1],
            'left': dir_path + '/IMG/' + line[1].split('/')[-1],
            'right': dir_path + '/IMG/' + line[2].split('/')[-1],
            'steering': float(line[3]),
            'throttle': float(line[4]),
            'brake': float(line[5]),
            'speed': float(line[6])})
    return lines

def make_sample_list(y_name, data_ref, expand_data=False, expand_correction=0.0):
    output_x = []
    output_y = []
    for d in data_ref:
        output_x.append(d["center"])
        output_y.append(d[y_name])
        if expand_data:
            output_x.append(d["left"])
            output_y.append(d[y_name]+expand_correction)
            output_x.append(d["right"])
            output_y.append(d[y_name]-expand_correction)
    return list(zip(output_x, output_y))

def make_model(model):
    input_layers = [
        Cropping2D(cropping=((60,50),(20,20)), input_shape=(160,320,3)),
        Lambda(lambda x: 2*((x / 255.0) - 0.5)),
        Dropout(0.2)
    ]
    if (model == 'nvidia'):
        feature_layers = [
            Conv2D(24, strides=(2, 2), kernel_size=(5, 5), activation= "elu"),
            Conv2D(36, strides=(2, 2), kernel_size=(5, 5), activation= "elu"),
            Conv2D(48, strides=(1, 1), kernel_size=(5, 5), activation= "elu"),
            Conv2D(64, kernel_size=(3, 3), activation= "elu"), 
            Conv2D(64, kernel_size=(3, 3), activation= "elu"),
        ]

        classification_layers = [
            Flatten(),
            Dropout(0.2),
            Dense(100),
            Activation("elu"),
            Dropout(0.5),
            Dense(50),
            BatchNormalization(),
            Activation("elu"),
            Dense(10),
            Activation("elu"),
            Dense(1)
        ]
    elif (model == 'reduced_nvidia'):
        feature_layers = [
            Conv2D(12, strides=(2, 2), kernel_size=(5, 5), activation= "elu"),
            Conv2D(18, strides=(2, 2), kernel_size=(5, 5), activation= "elu"),
            Conv2D(24, strides=(1, 1), kernel_size=(5, 5), activation= "elu"),
            Conv2D(32, kernel_size=(3, 3), activation= "elu"), 
            Conv2D(32, kernel_size=(3, 3), activation= "elu"),
        ]

        classification_layers = [
            Flatten(),
            Dropout(0.2),
            Dense(50),
            Activation("elu"),
            Dropout(0.5),
            Dense(25),
            BatchNormalization(),
            Activation("elu"),
            Dense(10),
            Activation("elu"),
            Dense(1)
        ]
    elif (model == 'lenet'):
        feature_layers = [
            Conv2D(12, kernel_size=(3, 3), padding="valid", activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2,2)),
            Conv2D(32, kernel_size=(3, 3), padding="valid", activation="relu"),
            MaxPooling2D(pool_size=(2, 2), strides=(2,2))
            ]

        classification_layers = [
            Flatten(),
            Dropout(0.5),
            Dense(120, activation="relu"),
            BatchNormalization(),
            Dense(84, activation="relu"),
            Dense(1)
        ]
    else:
        print('No valid model requested.')
        print('python model.py -h')
        sys.exit(2)
    return Sequential(input_layers + feature_layers + classification_layers)

def generator(samples, batch_size, data_augmentation=False):
    num_samples = len(samples)
    cols=320
    rows=160

    while 1:
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if (os.path.exists(batch_sample[0])):
                    if (batch_sample[0][0:10] == "IMG/center"):
                        if (abs(batch_sample[1])<0.01):
                            r =random.randint(0,400)
                            if (r<20):
                                img = plt.imread(batch_sample[0])
                                images.append(img)
                                angles.append(batch_sample[1])
                                images.append(np.fliplr(img))
                                angles.append(-batch_sample[1])
                            elif (data_augmentation):
                                r =random.randint(10,20)
                                img = plt.imread(batch_sample[0])
                                M = np.float32([[1,0, r],[0,1,0]])
                                images.append(cv2.warpAffine(img,M,(cols,rows)))
                                angles.append(batch_sample[1]+0.02*r)
                                
                                r =random.randint(-20,-10)
                                img = plt.imread(batch_sample[0])
                                M = np.float32([[1,0, r],[0,1,0]])
                                images.append(cv2.warpAffine(img,M,(cols,rows)))
                                angles.append(batch_sample[1]+0.02*r)
                    else:
                        img = plt.imread(batch_sample[0])
                        images.append(img)
                        angles.append(batch_sample[1])
                        images.append(np.fliplr(img))
                        angles.append(-batch_sample[1])
            X_train = np.array(images)
            Y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, Y_train)

def train_model(net_model, dataDir,outputfile):

    inputDirs = get_data_dirs(dataDir)
    if (len(inputDirs)==0):
        print("No data was found in {0}. Execution is stopped.", dataDir)
        sys.exit(2)

    print("{0} directory as input data found.".format(len(inputDirs)))
    for idir in inputDirs:
        print(idir)

    model = make_model(net_model)
    model.summary()
    plot_model(model, to_file=net_model+'_model_structure.png', show_shapes=True)

    batch_size = 32
    data_references = []
    for i in range(len(inputDirs)):
        data_dir = inputDirs[i]
        print("Start training with data from directory: ", data_dir)
        data_references += read_cvs(data_dir)

    train_data, valid_data = train_test_split(data_references,test_size=0.2)
    train_samples = make_sample_list("steering", train_data,expand_data=True, expand_correction=0.25)
    valid_samples = make_sample_list("steering", valid_data,expand_data=False)

    train_generator = generator(train_samples, batch_size=batch_size, data_augmentation=True)
    valid_generator = generator(valid_samples, batch_size=batch_size, data_augmentation=False)

    checkpoint_path = get_checkpoint_path("./")

    cp_callback = callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=1)
    es_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='min')

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                validation_data=valid_generator, validation_steps=math.ceil(len(valid_samples)/batch_size),
                epochs=10, verbose=1,
                callbacks=[cp_callback, es_callback])

    model.save(outputfile+'.h5')

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    plt.savefig(net_model+'_convergence_history.png')

def main(argv):
    model = ''
    inputfile = 'model_'
    outputfile = 'model_'
    dataDir = './Data/*'
    initial_learning = True
    transfer_learning = False
    try:
        opts, args = getopt.getopt(argv,"hd:m:",["ddir=","model="])
    except getopt.GetoptError:
        print ('python clone_behavior.py -h -t [on/off] -d <data_dir> -m model [lenet/nvidia/reduced_nvidia]')
        print ('-h: Write this help message.')
        print ('-m: model. Choice between lenet and nvidia.')
        print ('data_dir: Root directory for training data.  Default is ./Data/*' +
        ' Each of found directories need to contain driving_log.csv and a directory with images.')
        print ('Input and output files will have the names model_[lenet/nvidia]_i.h5, where i is an integer number.')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('python clone_behavior.py -h -t [on/off] -d <data_dir> -m model [lenet/nvidia/reduced_nvidia]')
            print ('-h: Write this help message.')
            print ('-m: model. Choice between lenet and nvidia.')
            print ('data_dir: Root directory for training data.  Default is ./Data/*' +
            ' Each of found directories need to contain driving_log.csv and a directory with images.')
            print ('Input and output files will have the names model_[lenet/nvidia]_i.h5, where i is an integer number.')
            sys.exit(0)
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-d", "--ddir"):
            dataDir = arg
        outputfile = 'model_' + model
    print('Training data will be taken from: ', dataDir)
    if transfer_learning:
        print('Transfer learning is requested with input from '+inputfile+'.h5')
    print('Output file is ', outputfile, ' +.h5')
    answer = input('Please confirm with "Y"!  :: ') 
    if (answer=="Y"):
        train_model(model,dataDir,outputfile)
    else:
        print("Input was not confirmed. Training will not start.")

if __name__ == "__main__":
    main(sys.argv[1:])
