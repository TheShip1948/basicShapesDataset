###############################################################################################
# Outline: 
# --------- 
# * Apply Dnn Mnist architecture  
# * Apply ConvNet Mnist architecture 
# TODO: 
# ----- 
# Enhance the image reading code 
###############################################################################################


###########################################
# --- Imports ---
###########################################
# from keras.datasets import cifar10
from keras.utils    		 import np_utils 
import numpy as np 
from keras.models   		 import Sequential 
from keras.layers   		 import Dense 
from keras.layers   		 import Dropout
from keras.layers.convolutional  import Convolution2D
from keras.layers.convolutional  import MaxPooling2D
from keras.layers		 import Flatten
import matplotlib.pyplot as plt 
import glob 
from PIL            		 import Image
from keras          		 import metrics
from skimage.transform 		 import resize

# For grid search in hyper-parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

###########################################
# --- Load data --- 
###########################################
# (X_train, y_train) , (X_test, y_test) = cifar10.load_data() 
"""
img = Image.fromarray(plt.imread("Data/TrainingSet/ClosedShapes/Circle/Cir1.bmp")).convert('1')
# img = img.convert('1')
# img = Image.open("Data/TrainingSet/ClosedShapes/Circle/Cir1.bmp").convert('1')
# img = 
plt.imshow(img)
plt.show()

fileList = glob.glob('Data/TrainingSet/ClosedShapes/Circle/*.bmp')

# print ("filelist: {}".format(filelist))

"""
"""
for i in range(0, 5):
	img = plt.imread(filelist[i])	
	plt.imshow(img)
	plt.show()
"""
"""
fileList = glob.glob('Data/TrainingSet/ClosedShapes/Circle/*.bmp') 
# Put in one numpy array 
X_train = np.array([np.array(Image.fromarray(plt.imread(fileName)).convert('1')) for fileName in fileList])

print("X_train shape: {}".format(X_train.shape))
"""


# --- --- Load Training Set --- --- 
def LoadPathImages(path): 
	fileList = glob.glob(path)
	dataArray = np.array([np.array(Image.fromarray(plt.imread(fileName)).convert('1')) for fileName in fileList])
	return dataArray 

# Shape - value dictionary 
shapeValueDict = {"Circle": 0,  "Diamond":1 ,"Ellipse":2 ,"Rectangle":3 , "Triangle":4 , "Arc":5 ,"Arrow":6 ,"Line":7 ,"Zigzag":8 }

# ---------------------
# --- Closed shapes --- 
# ---------------------
# Circle 
X_train_circle = LoadPathImages('Data/TrainingSet/ClosedShapes/Circle/*.bmp')
y_train_circle = np.full((X_train_circle.shape[0]), shapeValueDict['Circle']) 
print("X_train shape: {}".format(X_train_circle.shape)) 
print("y_train shape: {}".format(y_train_circle.shape)) 
print(y_train_circle[10])

# Diamond 
X_train_diamond = LoadPathImages('Data/TrainingSet/ClosedShapes/Diamond/*.bmp')
y_train_diamond = np.full((X_train_diamond.shape[0]), shapeValueDict['Diamond']) 
print("X_train shape: {}".format(X_train_diamond.shape)) 
print("y_train shape: {}".format(y_train_diamond.shape)) 
print(y_train_diamond[10])

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
print("Arc X_train shape {}".format(X_train_arc.shape))
print("Arc y_train shape {}".format(y_train_arc.shape))
 
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
# X_train_line = LoadPathImages_1('Data/TrainingSet/OpenShapes/Line/*.bmp')
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
# X_test_circle = LoadPathImages_1('Data/TestSet/ClosedShapes/Circle/*.bmp')
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

print("Stacking X_train shape: {}".format(X_train.shape))
print("Stacking y_train shape: {}".format(y_train.shape))

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

print("Stacking X_test shape: {}".format(X_test.shape))
print("Stacking y_test shape: {}".format(y_test.shape))

################################################
# --- Image Preprocessing --- 
################################################
# Resize images 
img_row_pixel_count = 50
img_col_pixel_count = 50

X_train_resized = np.array([np.array(resize(X_train[imageIndex], (img_row_pixel_count, img_col_pixel_count))) for imageIndex in range(X_train.shape[0])])
print("X_train_resized shape: {}".format(X_train_resized.shape))
X_test_resized = np.array([np.array(resize(X_test[imageIndex], (img_row_pixel_count, img_col_pixel_count))) for imageIndex in range(X_test.shape[0])])
print("X_test_resized shape: {}".format(X_test_resized.shape))


################################################
# --- Fixed seed number for reproducibility --- 
################################################
seed = 7 
np.random.seed(seed)

###########################################
# --- Flatten Input ---
###########################################
num_pixels = X_train_resized.shape[1]*X_train_resized.shape[2]
print ("num_pixels = {}".format(num_pixels))
X_train_resized = X_train_resized.reshape(X_train_resized.shape[0], num_pixels).astype('float32')
X_test_resized  = X_test_resized.reshape(X_test_resized.shape[0], num_pixels).astype('float32')
print("X_train_resized shape: {}".format(X_train_resized.shape))


###########################################
# --- Normalization --- 
###########################################
X_train_resized = X_train_resized.astype('float32')
X_test_resized  = X_test_resized.astype('float32')

X_train_resized = X_train_resized/255 
X_test_resized  = X_test_resized/255 

# Shift around zero 
# X_train_resized = X_train_resized + 0.5 
# X_test_resized  = X_test_resized + 0.5 


###########################################
# --- One hot encoding --- 
###########################################
y_train     = np_utils.to_categorical(y_train)
y_test      = np_utils.to_categorical(y_test) 
num_classes = y_test.shape[1]
print("num classes: {}".format(num_classes))


###########################################
# --- Define baseline model ---
###########################################
def baseline_model(init='normal', optimizer='adam'): 
	# Create model 
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init=init, activation='relu'))
	model.add(Dense(2000, input_dim=num_pixels, init=init, activation='relu'))
	#model.add(Dense(50,  init='normal', activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(500, input_dim=num_pixels, init='normal', activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(200, input_dim=num_pixels, init='normal', activation='relu'))
	# model.add(Dropout(0.2))	
	# model.add(Dense(num_classes, init='normal', activation='softmax'))
	# model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu'))
	# model.add(Convolution2D(32, 3, 3, input_shape=(1, 50, 50), border_mode='same', activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Convolution2D(32, 3, 3, input_shape=(1, 50, 50), border_mode='same', activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th'))
	# model.add(Flatten())
	# model.add(Dense(512, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model 
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy) 
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 	
	return model


###########################################
# --- Build the model ---
###########################################
# model = baseline_model() 
model = KerasClassifier(build_fn=baseline_model, verbose=0)

# Grid search parameters initialization 
# init = ['normal' , 'uniform', 'he_normal', 'zeros', 'ones']
init = ['normal']
batch_size = [50]
nb_epoch = [10] 
optimizer = ['adam', 'Nadam', 'RMSprop']
param_grid = dict(init=init, batch_size=batch_size, nb_epoch=nb_epoch, optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid.n_splits_=2 
# print("Split count: {}".format(grid.n_splits_)) 
###########################################
# --- Fit the model ---
###########################################
# model.fit(X_train_resized, y_train, validation_data=(X_test_resized, y_test), nb_epoch=100, batch_size=32, verbose=2)
# grid_result = grid.fit(X_train_resized, y_train, validation_data=(X_test_resized, y_test), nb_epoch=2, batch_size=32, verbose=2)
# grid_result = grid.fit(X_train_resized, y_train, nb_epoch=5, batch_size=50, verbose=1)
X = np.vstack((X_train_resized, X_test_resized))
y = np.concatenate((y_train, y_test))

print("X shape: {}".format(X.shape))
print("y shape: {}".format(y.shape)) 

# grid_result = grid.fit(X_train_resized[0:3000], y_train[0:3000], nb_epoch=1, verbose=1)
grid_result = grid.fit(X, y, verbose=1) 

# print("grid results: {}".format(dir(grid_result)))
# print("Split count: {}".format(grid_result.n_splits_))

###########################################
# --- Final evaluation ---
###########################################
# scores = model.evaluate(X_test_resized, y_test, verbose=0) 
# print('Log: score = {} %'.format(scores))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
	print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


