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
from keras.utils    import np_utils 
import numpy as np 
from keras.models   import Sequential 
from keras.layers   import Dense 
import matplotlib.pyplot as plt 
import glob 
from PIL            import Image


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


"""
###########################################
# --- Fixed seed number for reproducibility --- 
###########################################
print('Log: start seed definition')
seed = 7 
numpy.random.seed(seed)
print('Log: end seed definition')


###########################################
# --- Convert color images into gray ones --- 
###########################################
"""
"""
print('Log: start training images conversion')
timer.StartTime()
X_train_gray = numpy.ndarray(shape=(32, 32))
# for imageIndex in range(0, X_train.shape[0]): 
# TODO: handling the dimension of the numpy array is pretty bad, find a better solution 
# TODO: numpy create an initial record, I need to remove it 
# TODO: modify the sample size to 10000
X_training_sample_size = 10
for imageIndex in range(0, X_training_sample_size): 
	img = Image.fromarray(X_train[imageIndex])
	img = img.convert('1')
	X_train_gray = numpy.dstack([X_train_gray, img])
	print("Log: training image number = {}".format(imageIndex))
X_train_gray = numpy.swapaxes(X_train_gray, 1, 2)
X_train_gray = numpy.swapaxes(X_train_gray, 0, 1)
timer.EndTime()
timer.DeltaTime() 
print('Log: end training image conversion') 
print('Log: start testing image conversion')
timer.StartTime()
X_test_gray = numpy.ndarray(shape=(32, 32))
#for imageIndex in range(0, X_test.shape[0]):
# TODO: apply training modifications here
# TODO: code is similar may need to put in a function 
# TODO: the function may be generic enough to put outside the code 
# TODO: think of a library of utilities to be on git-hub   
for imageIndex in range(0, X_training_sample_size/5): 
	img = Image.fromarray(X_test[imageIndex])
	img = img.convert('1')
	X_test_gray = numpy.dstack([X_test_gray, img])
	print("Log: testing image number = {}".format(imageIndex))
X_test_gray = numpy.swapaxes(X_test_gray, 1, 2)	
X_test_gray = numpy.swapaxes(X_test_gray, 0, 1)
timer.EndTime()     
timer.DeltaTime()
print('Log: end testing image conversion')
print('Log: start input manipulation')
timer.StartTime()
print('Log: X_test_gray = {}'.format(X_test_gray))
"""

"""
###########################################
# --- Extract a sample ---
###########################################
training_sample_size = 10000
X_train = X_train[0:training_sample_size]
y_train = y_train[0:training_sample_size]

X_test  = X_test[0:training_sample_size/5]
y_test  = y_test[0:training_sample_size/5]


###########################################
# --- Flatten Input ---
###########################################
num_pixels = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
print ("num_pixels = {}".format(num_pixels))
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test  = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

###########################################
# --- Normalization --- 
###########################################

X_train = X_train/255 
X_test  = X_test/255 


###########################################
# --- One hot encoding --- 
###########################################
y_train     = np_utils.to_categorical(y_train)
y_test      = np_utils.to_categorical(y_test) 
num_classes = y_test.shape[1]

###########################################
# --- Define baseline model ---
###########################################
def baseline_model(): 
	# Create model 
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	# Compile model 
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
	return model


###########################################
# --- Build the model ---
###########################################
model = baseline_model() 


###########################################
# --- Fit the model ---
###########################################
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)


###########################################
# --- Final evaluation ---
###########################################
scores = model.evaluate(X_test, y_test, verbose=0) 
print('Log: score = {} %'.format(scores))
"""
