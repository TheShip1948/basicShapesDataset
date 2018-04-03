###############################################################################################
# Outline: 
# --------- 
# 
###############################################################################################


###########################################
# --- Imports ---
###########################################
from keras.utils    		 import np_utils 
import numpy as np 
from keras.models   		 import Sequential 
from keras.layers   		 import Dense 
from keras.layers   		 import Dropout
from keras.layers.convolutional  import Convolution2D
from keras.layers.convolutional  import MaxPooling2D
from keras.layers		 import Flatten
from keras.preprocessing.image   import ImageDataGenerator
from keras.preprocessing         import image 
from keras                       import applications 
import matplotlib.pyplot as plt 
import glob 
from PIL            		 import Image
from keras          		 import metrics
from skimage.transform 		 import resize, downscale_local_mean


###########################################
# --- Load data --- 
###########################################
# --- --- Load Training Set --- --- 
def LoadPathImages(path): 
	fileList = glob.glob(path)
	dataArray = np.array([np.array(image.img_to_array(image.load_img(fileName))) for fileName in fileList])
	return dataArray 

# Shape - value dictionary 
shapeValueDict = {"Circle": 0,  "Diamond":1 ,"Ellipse":2 ,"Rectangle":3 , "Triangle":4 , "Arc":5 ,"Arrow":6 ,"Line":7 ,"Zigzag":8 }

# ---------------------
# --- Closed shapes --- 
# ---------------------
# Circle 
X_train_circle = LoadPathImages('Data/TrainingSet/ClosedShapes/Circle/*.bmp')
y_train_circle = np.full((X_train_circle.shape[0]), shapeValueDict['Circle']) 

# Diamond 
X_train_diamond = LoadPathImages('Data/TrainingSet/ClosedShapes/Diamond/*.bmp')
y_train_diamond = np.full((X_train_diamond.shape[0]), shapeValueDict['Diamond']) 

# Ellipse 
X_train_ellipse = LoadPathImages('Data/TrainingSet/ClosedShapes/Ellipse/*.bmp')
y_train_ellipse = np.full((X_train_ellipse.shape[0]), shapeValueDict['Ellipse']) 

# Rectangle 
X_train_rectangle = LoadPathImages('Data/TrainingSet/ClosedShapes/Rectangle/*.bmp')
y_train_rectangle = np.full((X_train_rectangle.shape[0]), shapeValueDict['Rectangle']) 

# Triangle 
X_train_triangle = LoadPathImages('Data/TrainingSet/ClosedShapes/Triangle/*.bmp')
y_train_triangle = np.full((X_train_triangle.shape[0]), shapeValueDict['Triangle']) 

# -------------------
# --- Open shapes --- 
# ------------------- 
# Arc
X_train_arc = LoadPathImages('Data/TrainingSet/OpenShapes/Arc/*.bmp')
y_train_arc = np.full((X_train_arc.shape[0]), shapeValueDict['Arc']) 
 
# Arrow 
X_train_arrow = LoadPathImages('Data/TrainingSet/OpenShapes/Arrow/*.bmp')
y_train_arrow = np.full((X_train_arrow.shape[0]), shapeValueDict['Arrow']) 

# Line 
# TODO: remove this function, it is only for the purpose of debugging 
# TODO: Not all the images have the same number of color channels, ask Wael if this is intended or a bug? 
def LoadPathImages_1(path): 
	fileList = glob.glob(path)
	# dataArray = np.array([np.array(plt.imread(fileName)) for fileName in fileList])
	for fileName in fileList:	
		npArray = np.array(plt.imread(fileName))
		print("array shape: {},  fileName: {}".format(npArray.shape, fileName))
	return dataArray 

X_train_line = LoadPathImages('Data/TrainingSet/OpenShapes/Line/*.bmp')
y_train_line = np.full((X_train_line.shape[0]), shapeValueDict['Line']) 

# Zigzag
X_train_zigzag = LoadPathImages('Data/TrainingSet/OpenShapes/Zigzag/*.bmp')
y_train_zigzag = np.full((X_train_zigzag.shape[0]), shapeValueDict['Zigzag']) 

# --- --- Load Test Set --- --- 

# ---------------------
# --- Closed shapes --- 
# ---------------------
# Circle 
X_test_circle = LoadPathImages('Data/TestSet/ClosedShapes/Circle/*.bmp')
y_test_circle = np.full((X_test_circle.shape[0]), shapeValueDict['Circle']) 

# Diamond 
X_test_diamond = LoadPathImages('Data/TestSet/ClosedShapes/Diamond/*.bmp')
y_test_diamond = np.full((X_test_diamond.shape[0]), shapeValueDict['Diamond']) 

# Ellipse 
X_test_ellipse = LoadPathImages('Data/TestSet/ClosedShapes/Ellipse/*.bmp')
y_test_ellipse = np.full((X_test_ellipse.shape[0]), shapeValueDict['Ellipse']) 

# Rectangle 
X_test_rectangle = LoadPathImages('Data/TestSet/ClosedShapes/Rectangle/*.bmp')
y_test_rectangle = np.full((X_test_rectangle.shape[0]), shapeValueDict['Rectangle']) 

# Triangle 
X_test_triangle = LoadPathImages('Data/TestSet/ClosedShapes/Triangle/*.bmp')
y_test_triangle = np.full((X_test_triangle.shape[0]), shapeValueDict['Triangle']) 

# -------------------
# --- Open shapes --- 
# ------------------- 
# Arc
X_test_arc = LoadPathImages('Data/TestSet/OpenShapes/Arc/*.bmp')
y_test_arc = np.full((X_test_arc.shape[0]), shapeValueDict['Arc']) 
 
# Arrow 
X_test_arrow = LoadPathImages('Data/TestSet/OpenShapes/Arrow/*.bmp')
y_test_arrow = np.full((X_test_arrow.shape[0]), shapeValueDict['Arrow']) 

# Line 
X_test_line = LoadPathImages('Data/TestSet/OpenShapes/Line/*.bmp')
y_test_line = np.full((X_test_line.shape[0]), shapeValueDict['Line']) 

# Zigzag
X_test_zigzag = LoadPathImages('Data/TestSet/OpenShapes/Zigzag/*.bmp')
y_test_zigzag = np.full((X_test_zigzag.shape[0]), shapeValueDict['Zigzag']) 

####################################################
# --- Compile training data into single array  --- 
####################################################
training_sample_size = 249

# --- Closed shapes --- 
X_train = np.vstack((X_train_circle[0:training_sample_size], X_train_diamond[0:training_sample_size]))
y_train = np.concatenate((y_train_circle[0:training_sample_size], y_train_diamond[0:training_sample_size]))

X_train = np.vstack((X_train, X_train_ellipse[0:training_sample_size]))
y_train = np.concatenate((y_train, y_train_ellipse[0:training_sample_size]))

X_train = np.vstack((X_train, X_train_rectangle[0:training_sample_size]))
y_train = np.concatenate((y_train, y_train_rectangle[0:training_sample_size]))

X_train = np.vstack((X_train, X_train_triangle[0:training_sample_size]))
y_train = np.concatenate((y_train, y_train_triangle[0:training_sample_size]))

# --- Open shapes --- 
X_train = np.vstack((X_train, X_train_arc[0:training_sample_size]))
y_train = np.concatenate((y_train, y_train_arc[0:training_sample_size]))

X_train = np.vstack((X_train, X_train_arrow[0:training_sample_size]))
y_train = np.concatenate((y_train, y_train_arrow[0:training_sample_size]))

X_train = np.vstack((X_train, X_train_line[0:training_sample_size]))
y_train = np.concatenate((y_train, y_train_line[0:training_sample_size]))

X_train = np.vstack((X_train, X_train_zigzag[0:training_sample_size]))
y_train = np.concatenate((y_train, y_train_zigzag[0:training_sample_size]))

####################################################
# --- Compile testing data into single array  --- 
####################################################
testing_sample_size  = 122

# --- Closed shapes --- 
X_test = np.vstack((X_test_circle[0:testing_sample_size], X_test_diamond[0:testing_sample_size]))
y_test = np.concatenate((y_test_circle[0:testing_sample_size], y_test_diamond[0:testing_sample_size]))

X_test = np.vstack((X_test, X_test_ellipse[0:testing_sample_size]))
y_test = np.concatenate((y_test, y_test_ellipse[0:testing_sample_size]))

X_test = np.vstack((X_test, X_test_rectangle[0:testing_sample_size]))
y_test = np.concatenate((y_test, y_test_rectangle[0:testing_sample_size]))

X_test = np.vstack((X_test, X_test_triangle[0:testing_sample_size]))
y_test = np.concatenate((y_test, y_test_triangle[0:testing_sample_size]))

# --- Open shapes --- 
X_test = np.vstack((X_test, X_test_arc[0:testing_sample_size]))
y_test = np.concatenate((y_test, y_test_arc[0:testing_sample_size]))

X_test = np.vstack((X_test, X_test_arrow[0:testing_sample_size]))
y_test = np.concatenate((y_test, y_test_arrow[0:testing_sample_size]))

X_test = np.vstack((X_test, X_test_line[0:testing_sample_size]))
y_test = np.concatenate((y_test, y_test_line[0:testing_sample_size]))

X_test = np.vstack((X_test, X_test_zigzag[0:testing_sample_size]))
y_test = np.concatenate((y_test, y_test_zigzag[0:testing_sample_size]))

################################################
# --- Image Preprocessing --- 
################################################
# Resize images 
img_row_pixel_count = 50
img_col_pixel_count = 50

X_train_resized = np.array([np.array(downscale_local_mean(X_train[imageIndex], (5, 5, 1))) for imageIndex in range(X_train.shape[0])])
X_test_resized = np.array([np.array(downscale_local_mean(X_test[imageIndex], (5, 5, 1))) for imageIndex in range(X_test.shape[0])])

################################################
# --- Fixed seed number for reproducibility --- 
################################################
seed = 7 
np.random.seed(seed)


###########################################
# --- Convert to float  --- 
###########################################
X_train_resized = X_train_resized.astype('float32')
X_test_resized  = X_test_resized.astype('float32')

##########################################
# --- One hot encoding --- 
###########################################
y_train     = np_utils.to_categorical(y_train)
y_test      = np_utils.to_categorical(y_test) 
num_classes = y_test.shape[1]

# --- Reshape input --- 
X_train_resized = X_train_resized.reshape(X_train_resized.shape[0], 3, X_train_resized.shape[1], X_train_resized.shape[2])
X_test_resized = X_test_resized.reshape(X_test_resized.shape[0], 3, X_test_resized.shape[1], X_test_resized.shape[2])

###########################################
# --- Use Pretrained Model ---
###########################################
def save_bottleneck_features(X_train_resized, y_train, X_test_resized, y_test):
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # Parameters 
    batch_size= 32

    # Generate and save training data activation maps 	
    X_train_resized = X_train_resized.reshape(X_train_resized.shape[0], X_train_resized.shape[2], X_train_resized.shape[3], 3)
    bottleneck_features_train = model.predict(X_train_resized, batch_size= batch_size, verbose=1)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    # Generate and save test data activation maps 
    X_test_resized = X_test_resized.reshape(X_test_resized.shape[0], X_test_resized.shape[2], X_test_resized.shape[3], 3)
    bottleneck_features_validation = model.predict(X_test_resized, batch_size= batch_size, verbose=1)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = y_train 

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = y_test

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=100,
              batch_size=16,
              validation_data=(validation_data, validation_labels))

# save_bottleneck_features(X_train_resized, y_train, X_test_resized, y_test)
train_top_model()




