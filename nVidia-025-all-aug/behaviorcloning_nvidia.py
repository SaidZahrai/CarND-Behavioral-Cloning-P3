import glob
import os
import csv
import cv2
import sys, getopt
import datetime

import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential, save_model, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, MaxPooling2D, Activation, Cropping2D
from keras.utils import plot_model
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

def make_model():
    input_layers = [
        Cropping2D(cropping=((60,50),(20,20)), input_shape=(160,320,3)),
        Lambda(lambda x: ((x / 255.0) - 0.5))
    ]

    feature_layers = [
        Conv2D(12, strides=(2, 2), kernel_size=(5, 5), activation='relu'),
        Conv2D(18, strides=(2, 2), kernel_size=(5, 5), activation='relu'),
#        Conv2D(24, strides=(2, 2), kernel_size=(5, 5), activation='relu'),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        Conv2D(32, kernel_size=(3, 3), activation='relu')
        ]

    classification_layers = [
        Flatten(),
        Dropout(0.2),
        Dense(100, activation="relu"),
        Dropout(0.5),
        Dense(50, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="relu"),
        Dropout(0.5),
        Dense(1)
    ]

    return Sequential(input_layers + feature_layers + classification_layers)

def generator(samples, batch_size):
    num_samples = len(samples)
    cols=320
    rows=160

    M_toright = np.float32([[1,0, 20],[0,1,0]])
    M_toleft  = np.float32([[1,0,-20],[0,1,0]])

    while 1:
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if (os.path.exists(batch_sample[0])):
                    img = plt.imread(batch_sample[0])
                    images.append(img)
                    angles.append(batch_sample[1])
                    images.append(np.fliplr(img))
                    angles.append(-batch_sample[1])
                    images.append(cv2.warpAffine(img,M_toright,(cols,rows)))
                    angles.append(batch_sample[1]-0.1)
                    images.append(cv2.warpAffine(img,M_toleft,(cols,rows)))
                    angles.append(batch_sample[1]+0.1)
            X_train = np.array(images)
            Y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, Y_train)

def do_initial_learning(dataDir,outputfile):

    inputDirs = get_data_dirs(dataDir+"/*")
    if (len(inputDirs)==0):
        print("No data was found in {0}. Execution is stopped.", dataDir)
        sys.exit(2)

    print("{0} directory as input data found.".format(len(inputDirs)))
    for idir in inputDirs:
        print(idir)

    model = make_model()
    model.summary()
    plot_model(model, to_file='nvidia_model_structure.png', show_shapes=True)

    batch_size = 32
    data_references = []
    for i in range(len(inputDirs)):
        data_dir = inputDirs[i]
        print("Start training with data from directory: ", data_dir)
        data_references += read_cvs(data_dir)

    train_data, valid_data = train_test_split(data_references,test_size=0.2)
    train_samples = make_sample_list("steering", train_data,expand_data=True, expand_correction=0.25)
    valid_samples = make_sample_list("steering", valid_data,expand_data=False)

    train_generator = generator(train_samples, batch_size=batch_size)
    valid_generator = generator(valid_samples, batch_size=batch_size)

    checkpoint_path = get_checkpoint_path("Training")

    cp_callback = callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=1)
    es_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='min')

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                validation_data=valid_generator, validation_steps=math.ceil(len(valid_samples)/batch_size),
                epochs=5, verbose=1,
                callbacks=[cp_callback, es_callback])

    model.save(outputfile+'0.h5')

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
    plt.savefig('convergence_history.png')


def do_transfer_learning(dataDir,model_file,outputfile):

    inputDirs = get_data_dirs(dataDir+"/test*")
    if (len(inputDirs)==0):
        print("No data was found in {0}. Execution is stopped.".format(dataDir))
        sys.exit(2)

    model_file += '.h5'
    if (not os.path.exists(model_file)):
        print("Model file {0} was not found. Execution is stopped.".format(model_file))
        sys.exit(2)

    print("{0} director(y/ies) as input data found.".format(len(inputDirs)))
    for idir in inputDirs:
        print(idir)

    model = load_model(model_file)
    model.get_layer("conv2d_1").trainable=False
    model.get_layer("conv2d_2").trainable=False
    model.summary()
    plot_model(model, to_file='transfer_model_structure.png', show_shapes=True)

    batch_size = 32

    for i in range(len(inputDirs)):
        data_dir = inputDirs[i]
        print("Start training with data from directory: ", data_dir)
        data_references = read_cvs(data_dir)

        train_data, valid_data = train_test_split(data_references,test_size=0.2)
        train_samples = make_sample_list("steering", train_data,expand_data=True, expand_correction=0.2)
        valid_samples = make_sample_list("steering", valid_data,expand_data=False)

        train_generator = generator(train_samples, batch_size=batch_size)
        valid_generator = generator(valid_samples, batch_size=batch_size)

        checkpoint_path = get_checkpoint_path("Training")

        cp_callback = callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=1)
        es_callback = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=1, mode='min')

        model.compile(loss='mse', optimizer='adam')
        model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                    validation_data=valid_generator, validation_steps=math.ceil(len(valid_samples)/batch_size),
                    epochs=5, verbose=1,
                    callbacks=[cp_callback, es_callback])
                    
        model.save(outputfile+str(i+1)+'.h5')

def main(argv):
    inputfile = 'model_nvidia_0'
    outputfile = 'model_nvidia_'
    dataDir = './Data'
    initial_learning = True
    transfer_learning = False
    try:
        opts, args = getopt.getopt(argv,"ht:d:i:o:",["ddir=","ifile=","ofile="])
    except getopt.GetoptError:
        print ('behaviourcloning_nvidia.py -h -t [0/1] -d <data_dir> -i <history> -o <outputfile>')
        print ('-h: Write this help message.')
        print ('-t: Transfer learning switch. off: do not perform, on: Only transffer learning.')
        print ('data_dir: Root directory for training data.  Default is ./Data' +
        ' Each of child directories need to contain driving_log.csv and a directory with images.')
        print ('history: Starting point for transfer learning.' +
        ' Extension h5 is added. Default is model_nvidia_.')
        print ('outputfile: Output model file basic name.' +
        ' For each of data directories, a file is saved with this name<counter>.h5.  Default is model_nvidia_.')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('behaviourcloning_nvidia.py -h -t [0/1] -d <data_dir> -i <history> -o <outputfile>')
            print ('-h: Write this help message.')
            print ('-t: Transfer learning switch. off: do not perform, on: Only transffer learning.')
            print ('data_dir: Root directory for training data.  Default is ./Data' +
            ' Each of child directories need to contain driving_log.csv and a directory with images.')
            print ('history: Starting point for transfer learning.' +
            ' Extension h5 is added. Default is model_nvidia_.')
            print ('outputfile: Output model file basic name.' +
            ' For each of data directories, a file is saved with this name<counter>.h5.  Default is model_nvidia_.')
            sys.exit(0)
        elif opt == '-t':
            print("-t: ", arg)
            if (arg == 'off'):
                initial_learning = True
                transfer_learning = False
            elif (arg == 'on'):
                initial_learning = False
                transfer_learning = True
            else:
                print('Incorrect argument for switch -t. Execution stopped.')
                sys.exit(2)
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-d", "--ddir"):
            dataDir = arg
    print('Training data will be taken from: ', dataDir)
    if transfer_learning:
        print('Transfer learning is requested with input from '+inputfile+'.h5')
    print('Output file is ', outputfile)
    answer = input('Please confirm with "Y"!  :: ') 
    if (not answer=="Y"):
        print("Inoput was not confirmed. Training will not start.")
    else:
        if (initial_learning):
            do_initial_learning(dataDir,outputfile)
        if (transfer_learning):
            do_transfer_learning(dataDir,inputfile,outputfile)

if __name__ == "__main__":
    main(sys.argv[1:])
