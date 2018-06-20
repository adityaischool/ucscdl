import tensorflow as tf
import numpy as np
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from mpl_toolkits.axes_grid1 import ImageGrid


# Create dictionary of target classes# Creat 
label_dict = {
 0: 'Bread',
 1: 'Dairy product',
 2: 'Dessert',
 3: 'Egg',
 4: 'Fried food',
 5: 'Meat',
 6: 'Noodles/Pasta',
 7: 'Rice',
 8: 'Seafood',
 9: 'Soup',
 10: 'Vegetable/Fruit',
}

cwd = os.getcwd()

## Place the data in Food-11 directory
data_in_dir = os.path.join(cwd, "Food-11")
assert(os.path.isdir(data_in_dir))

subdirs = {
    'train' : 'training',
    'valid' : 'validation',
    'eval'  : 'evaluation'}
dirs = os.listdir(data_in_dir)
print(dirs)
## Validate we have these 3 subdirectories
assert(len(dirs) == len(subdirs) and sorted(dirs) == sorted(subdirs.values()))

## Create output directory in current path to store images
image_dir = os.path.join(cwd, "food-classification-eda-images")
if not os.path.exists(image_dir): os.makedirs(image_dir)
    
## Create output directory to store the dataframes in pickle format
pickle_dir = os.path.join(cwd, "food-classification-pickle_data")
if not os.path.exists(pickle_dir): os.makedirs(pickle_dir)
## training, validation and evaluation data directories
training_data_dir = os.path.join(data_in_dir, subdirs['train'])
validation_data_dir = os.path.join(data_in_dir, subdirs['valid'])
evaluation_data_dir = os.path.join(data_in_dir, subdirs['eval'])

## training, validation and evaluation data images
training_images = glob.glob(os.path.join(training_data_dir, "*"))
validation_images = glob.glob(os.path.join(validation_data_dir, "*"))
evaluation_images = glob.glob(os.path.join(evaluation_data_dir, "*"))

all_images = [training_images, validation_images, evaluation_images]
all_directories = [training_data_dir, validation_data_dir, evaluation_data_dir]



# Prepare Training Dataframe

training = pd.DataFrame(training_images)
training.columns = ['Path']
training['Label'] = training.Path.apply(lambda x: os.path.basename(x).split('_')[0])
training.describe()

#Validate a single training image

fig, ax = plt.subplots()
ix = np.random.randint(0, len(training)) # randomly select a index
img_path = training.Path[ix]
plt.imshow(io.imread(img_path), cmap='binary')
cat = training.Label[ix] # get the radiograph category
plt.title('Path: %s \n Label: %s(%s) ' %(os.path.basename(img_path), cat, label_dict[int(cat)]))
plt.show()
fig.savefig(os.path.join(image_dir, 'one_sample_image_test.jpg'), bbox_inches='tight', pad_inches=0)

# Prepare Validation Dataframe

validation = pd.DataFrame(validation_images)

validation.columns = ['Path']
validation['Label'] = validation.Path.apply(lambda x: os.path.basename(x).split('_')[0])
validation.describe()

# Prepare Evaluation Dataframe


evaluation = pd.DataFrame(evaluation_images)

evaluation.columns = ['Path']
evaluation['Label'] = evaluation.Path.apply(lambda x: os.path.basename(x).split('_')[0])
evaluation.describe()

### ONE TIME
training.to_pickle(os.path.join(pickle_dir, "training.pickle"))
validation.to_pickle(os.path.join(pickle_dir, "validation.pickle"))
evaluation.to_pickle(os.path.join(pickle_dir, "evaluation.pickle"))