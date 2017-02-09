#Main file for training the network for Behavioral Cloning Project - Project 3 of Udacity Self Driving Car Nanodegree
#The code is developed entirely using ipython notebook
#It has been copy-pasted to this file as suggested in project submission guidelines


#Get Data
import os
import numpy as np
import csv
from scipy.misc import imread
from random import shuffle

########Set these things before running the script########
#Bunch of things which need to be set

headers = ["center","left","right","steering"]
root = os.getcwd() + "/Data"
paths_to_data = ['/BaseData/']
model_save_filename = './Models/Nvidia_architecture_center_left_right_images_flipCenter_aug_7Epoch_val_0_0132.h5'


#Split ratio for splitting data into Training and Validation Data Sets
train_split = 0.8
val_split = 0.2

#Train config
BATCH_SIZE = 128
EPOCHS = 5

#########################################################


#With the given paths to data, get the image location from teh corresponding csv log file..
#and also the targetvalue(steeringAngle in our present case), and store them.
##Only the image locations are being accessed. The image themselves will be individually accessed during train time..
#using generators to satisfy memory constraints.

data = []
for path in paths_to_data:
    global_path = root + path
    path_to_csv = root + path + 'driving_log.csv'
    
    with open(path_to_csv) as csv_file:
        csvreader = csv.reader(csv_file, skipinitialspace=True)
        next(csvreader,None) # skip header row
        for row in csvreader:
            data.append(((global_path+row[0], global_path+row[1], global_path+row[2], row[3])))


#Functions to get image and target values given the path
#Used by train and validation generators

#Get image as a numpy array
def get_image (image_path):
    return np.array([imread(image_path)]).astype(np.float32)   

#Get steering value - function to faciliate usage of generators
def get_value (steering_value):
    return np.array([float(steering_value)])

#Flip images and adjust steering
def get_flipped_image (image_path):
    image_array = np.array([imread(image_path)]).astype(np.float32)
    return np.fliplr(image_array)            


#Class for infintely recurring generator object 
class InfiniteRecurGenerator:
    def __init__(self,data,train_split,val_split):
        self.data = data
        self.train_split = train_split
        self.val_split = val_split
        self.shuffle_data()
        self.split_data()
        self.train_generator = self.get_generator(self.train_data)
        self.val_generator = self.get_generator(self.val_data)
    
    #Shuffle data
    def shuffle_data(self):
        shuffle(self.data)
    
    def shuffle_train_data(self):
        shuffle(self.train_data)
    
    def shuffle_val_data(self):
        shuffle(self.val_data)
        
    #Split data into train and validation sets
    def split_data(self):
        self.num_of_examples = len(self.data)
        #self.num_train_examples = math.floor(self.num_of_examples * self.train_split)
        #self.num_val_examples = math.floor(self.num_of_examples * self.val_split)
       
        #For python 2
        self.num_train_examples = int(self.num_of_examples * self.train_split)
        self.num_val_examples = int(self.num_of_examples * self.val_split)

        self.train_data = self.data[0:self.num_train_examples]
        self.val_data = self.data[self.num_train_examples+1:self.num_train_examples+self.num_val_examples]

    #Create data generators
    def get_generator(self,data_for_gen):
        for center, left, right, steering in data_for_gen:
            yield (get_image(center), get_value(steering))
            yield (get_image(left), get_value(float(steering) + 0.2))
            yield (get_image(right), get_value(float(steering) - 0.2))
            yield (get_flipped_image(center),get_value(steering)* -1)
            

    #Both these Functions transfered to init itself
    
    #set train generator    
    #def set_train_generator(self):
    #    self.train_generator = self.get_generator(self.train_data)
    
    #set validation generator 
    #def set_val_generator(self):
    #    self.val_generator = self.get_generator(self.val_data)
    
    #to be used by the keras model as a generator for fetching examples for training    
    def get_train_example(self):
        while True:
            try:
                yield next(self.train_generator)
            except StopIteration:
                self.shuffle_train_data()
                #self.split_data()
                self.train_generator =  self.get_generator(self.train_data)
                yield next(self.train_generator)
    
    #to be used by the keras model as a generator for fetching examples for validation
    def get_val_example(self):
        while True:
            try:
                yield next(self.val_generator)
            except StopIteration:
                self.shuffle_val_data
                #self.split_data()
                self.val_generator = self.get_generator(self.val_data)
                yield next(self.val_generator)



#Build the Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU,ELU

def crop(x):
    import tensorflow as tf
    #Remove top 1/3rd of the image (sky part of the graphic as it doesn't add any value in training)
    #Remove the bottom 25 pixels to remove the bonnet of the car
    #Both methods succesfully employed in Project1 - Lane Finding
    x =  x[:, 60:135, 0:320]
    #resize the images to 66*200 (height * width) to keep in tune with the nVIDIA DAVE2 architecture
    x = tf.image.resize_images(x, [66, 200])
    return x

def normalize(x):
    #normalize the image values - best practices used for neural net training
    return x / 127.5 - 1


def nvidia_architecture():
    #Define nVIDIA DAVE2 architecture
    model = Sequential()
    
    #Preprocessing of the images
    model.add(Lambda(crop, input_shape=(160, 320, 3), name="crop"))
    model.add(Lambda(normalize, name="normalize"))
    
    #First convolution layer
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid",init='he_normal'))
    model.add(ELU())
    
    #Second convolution layer
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid",init='he_normal'))
    model.add(ELU())
    
    #Third convolution layer
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid",init='he_normal'))
    model.add(ELU())
    
    #Fourth convolution layer - unit striding as suggested in nVIDIA DAVE2 paper
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid",init='he_normal'))
    model.add(ELU())
    
    #Fifth convolution layer - unit striding as suggested in nVIDIA DAVE2 paper
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid",init='he_normal'))
    model.add(ELU())
    
    model.add(Flatten())
    
    #model.add(Dense(1164, init='he_normal'))
    #model.add(ELU())
    
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    
    model.add(Dense(1, init='he_normal'))
    
    return model


############
#Training process
############

model = nvidia_architecture()
#model.summary()

#The generators need to be infintely looping
data_generator = InfiniteRecurGenerator(data,train_split,val_split)

#Compile the model
model.compile(loss='mse', optimizer=Adam(lr=0.0001))

#Train the model
model.fit_generator(generator= data_generator.get_train_example(),samples_per_epoch = data_generator.num_train_examples * 4, nb_epoch = EPOCHS ,
                    validation_data = data_generator.get_val_example(), verbose=1, nb_val_samples = data_generator.num_val_examples * 4)

#Save the model
model.save('./Models/Nvidia_architecture_center_left_right_images_flipCenter_aug_7Epoch_val_0_0132.h5')
