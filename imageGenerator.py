###############################################################################################
# Goal: 
# --------- 
# * Implement image generator 
###############################################################################################


###########################################
# --- Imports ---
###########################################

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob 
from PIL            		 import Image
import numpy as np  
import matplotlib.pyplot as plt  
# from skimage.color import rgba2rgb 
from skimage.transform 		 import resize 
from skimage.transform 		 import downscale_local_mean
###########################################
# --- Implementation ---
###########################################
"""
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# img = load_img('Data/TrainingSet/ClosedShapes/Circle/Cir1.bmp')  # this is a PIL image
img = load_img('Data/TrainingSet/ClosedShapes/Circle/Cir6.bmp')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
print("image shape = {}".format(x.shape))
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='AugmentedData/TrainingSet/ClosedShapes/Circle/', save_prefix='Cir', save_format='bmp'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

"""


# Load images in a path, convert them to grayscale images and return numpy array 
def LoadPathImages(path): 
	fileList = glob.glob(path)
	# dataArray = np.array([np.array(Image.fromarray(plt.imread(fileName)).convert('1')) for fileName in fileList])
	# dataArray = np.array([np.array(Image.fromarray(plt.imread(fileName))) for fileName in fileList])
	dataArray = np.array([np.array(img_to_array(load_img(fileName))) for fileName in fileList])
	return dataArray  

# Image generation  
def GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage): 
	# Image generator 
	datagen = ImageDataGenerator(
		zca_whitening=True, 
		rotation_range=40,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode='nearest')

	# Load images from path 
	imgArray = LoadPathImages(loadPath)	

	# Generate images and save in save path 
	for imgIndex in range(0, imgArray.shape[0]):	
		i = 0
		img = imgArray[imgIndex]
		print("img shape = {}".format(img.shape))	
		img = img.reshape((1,) + img.shape)
		print('image shape = {}'.format(img.shape))
		for batch in datagen.flow(img, batch_size=1,
				          save_to_dir=savePath, save_prefix=savePrefix, save_format='bmp'):
		    i += 1
		    if i > generatedImageCountPerRealImage:
			break  # otherwise the generator would loop indefinitely
	  


###########################################
# --- Generate Closed Shapes ---
###########################################
# Generate circle images 
loadPath = 'Data/TrainingSet/ClosedShapes/Circle/*.bmp' 
savePath = 'AugmentedData/TrainingSet/ClosedShapes/Circle/' 
savePrefix = 'Circle' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)

# Generate diamond images 
loadPath = 'Data/TrainingSet/ClosedShapes/Diamond/*.bmp' 
savePath = 'AugmentedData/TrainingSet/ClosedShapes/Diamond/' 
savePrefix = 'Diamond' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)

# Generate ellipse images 
loadPath = 'Data/TrainingSet/ClosedShapes/Ellipse/*.bmp' 
savePath = 'AugmentedData/TrainingSet/ClosedShapes/Ellipse/' 
savePrefix = 'Ellipse' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)

# Generate rectangle images 
loadPath = 'Data/TrainingSet/ClosedShapes/Rectangle/*.bmp' 
savePath = 'AugmentedData/TrainingSet/ClosedShapes/Rectangle/' 
savePrefix = 'Rectangle' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)

# Generate triangle images 
loadPath = 'Data/TrainingSet/ClosedShapes/Triangle/*.bmp' 
savePath = 'AugmentedData/TrainingSet/ClosedShapes/Triangle/' 
savePrefix = 'Triangle' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)

###########################################
# --- Generate Open Shapes ---
###########################################
# Generate arc images 
loadPath = 'Data/TrainingSet/OpenShapes/Arc/*.bmp' 
savePath = 'AugmentedData/TrainingSet/OpenShapes/Arc/' 
savePrefix = 'Arc' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)


# Generate arrow images 
loadPath = 'Data/TrainingSet/OpenShapes/Arrow/*.bmp' 
savePath = 'AugmentedData/TrainingSet/OpenShapes/Arrow/' 
savePrefix = 'Arrow' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)


# Generate line images 
loadPath = 'Data/TrainingSet/OpenShapes/Line/*.bmp' 
savePath = 'AugmentedData/TrainingSet/OpenShapes/Line/' 
savePrefix = 'Line' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)


# Generate zigzag images 
loadPath = 'Data/TrainingSet/OpenShapes/Zigzag/*.bmp' 
savePath = 'AugmentedData/TrainingSet/OpenShapes/Zigzag/' 
savePrefix = 'Zigzag' 
generatedImageCountPerRealImage = 20
GenerateImages(loadPath, savePath, savePrefix, generatedImageCountPerRealImage)




