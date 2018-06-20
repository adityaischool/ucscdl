import tensorflow as tf
import numpy as np
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import cv2


def imageToNp (path):
    imgnp = cv2.imread(path)#.flatten()
    imgnp = cv2.resize(imgnp, (128, 128)).flatten() 
    #imResize = imgnp.resize((128,128))
    #imResize=imResize.flatten();
    return imgnp;
#imageToNp (r"C:\Users\gkeshavd\Desktop\ML2\proj1\AI-Food-Classification-master\AI-Food-Classification-master\Food-11\training\\0_0.jpg")
def imagesToNp (paths ):
    con = []
    for op in paths :
        con.append( imageToNp(op).tolist());
    return np.array(con);

def get_one_hot(targets, nb_classes):
    #print (targets);
    return np.eye(nb_classes)[np.array(targets).astype(int).reshape(-1)].tolist()

def getbatch(xval, yval,  batchsize=100):
    arraylength = len (xval);
    #print(xval);
    count = 0 
    while count < arraylength/batchsize:
        randstart = random.randint(0, arraylength-batchsize-1)
        count += 1
        yield (  imagesToNp (xval[randstart:randstart+batchsize]) ,  get_one_hot(yval[randstart:randstart+batchsize],n_classes))
		
def conv2dnew(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
#returns RELU(convWX + B)

def maxpool2dnew(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

 
def conv_net(x, weightsnew, biasesnew, dropout):
    # reshape input to 128x128 size
    x = tf.reshape(x, shape=[-1, 128, 128, 3])

    # Convolution layer 1
    conv1 = conv2dnew(x, weightsnew['wc1'], biasesnew['bc1'])
    # Max pooling
    conv1 = maxpool2dnew(conv1, k=2)

    # Convolution layer 2
    conv2 = conv2dnew(conv1, weightsnew['wc2'], biasesnew['bc2'])
    # Max pooling
    conv2 = maxpool2dnew(conv2, k=2)

    # Fully connected layer
    fc1 = tf.reshape(conv2, [-1, weightsnew['wd1'].get_shape().as_list()[0]])
    #reshape it to one long layer... 
    fc1 = tf.add(tf.matmul(fc1, weightsnew['wd1']), biasesnew['bd1'])
    # this is WX + b
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weightsnew['out']), biasesnew['out']) # output layer
    return out
	



# Load data
cwd = os.getcwd()
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

## Place the data in Food-11 directory
data_in_dir = os.path.join(cwd, "Food-11")
assert(os.path.isdir(data_in_dir))

subdirs = {
    'train' : 'training',
    'valid' : 'validation',
    'eval'  : 'evaluation'}
dirs = os.listdir(data_in_dir)

## Validate we have these 3 subdirectories
assert(len(dirs) == len(subdirs) and sorted(dirs) == sorted(subdirs.values()))
   
## Validate that we have the sored data from EDA in pickle format
pickle_dir = os.path.join(cwd, "food-classification-pickle_data")
assert(os.path.isdir(pickle_dir))
data_files = os.listdir(pickle_dir)
data_files
datastore_files = {
    'train' : 'training.pickle',
    'valid' : 'validation.pickle',
    'eval'  : 'evaluation.pickle'}
## Validate we have these 3 datafiles
assert(len(data_files) == len(datastore_files) and sorted(data_files) == sorted(datastore_files.values()))

#Read data from pickle file to dataframes

train_data = pd.read_pickle(os.path.join(pickle_dir, datastore_files['train']))
val_data = pd.read_pickle(os.path.join(pickle_dir, datastore_files['valid']))
eval_data = pd.read_pickle(os.path.join(pickle_dir, datastore_files['eval']))

# architecture hyper-parameter
#noofdatapoints = mnist.train.num_examples

learningrate = 0.0005
nepochs = 100
batch_size = 100
noofbatches = 100#noofdatapoints//batch_size
print("no of batches =", noofbatches)

n_input = 49152 # 128x128x3 image
n_classes = 11 # 1 for each digit [0-9]
dropout = 0.75 

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
print(X.shape, Y.shape)

weightsnew = {
    # conv filters: choosing 5x5 and 3 channels but only 10 filters
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 10])),
    # filters for 2nd conv layer : 5x5 again but 20 filters
    # 3rd param is 10 because first conv layer output is 10 deep channels
    'wc2': tf.Variable(tf.random_normal([5, 5, 10, 20])),
    'wd1': tf.Variable(tf.random_normal([32*32*20, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}


biasesnew = {
    'bc1': tf.Variable(tf.random_normal([10])),
    'bc2': tf.Variable(tf.random_normal([20])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}  

# code same as multi layer perc
#Create the model
model = conv_net(X, weightsnew, biasesnew, keep_prob)
print(model)
# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
train_min = tf.train.AdamOptimizer(learning_rate=learningrate).minimize(loss)

# Evaluate model
correct_model = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_model, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(10):
        for _ in range(noofbatches):
            G= getbatch(train_data["Path"].values.tolist(),train_data["Label"].values.tolist(),batch_size)
            batch_x, batch_y= next(G);
            #print("before resizing", batch_x.shape)
            #batch_x=tf.image.resize_images(images=batch_x,size=[128,128])
            #print("After resizing", batch_x.shape)
            #batch_y = batch_y.astype(np.float32)
            # Use training data for optimization
            #_, costval, merged_summary = 
            sess.run(train_min, feed_dict={X:batch_x, Y:batch_y, keep_prob: dropout})
            #writer.add_summary(merged_summary, epoch)
        # Validate after every epoch
        batch_x, batch_y = next(getbatch(train_data["Path"].values.tolist(),train_data["Label"].values.tolist(),batch_size));
        losscalc, accuracycalc = sess.run([loss, accuracy], 
                                          feed_dict={X:batch_x, Y:batch_y, keep_prob: 1.0})
        print("Epoch: %d, Loss: %0.4f, Accuracy: %0.4f"%(epoch, losscalc, accuracycalc))
            
    # When the training is complete and you are happy with the result
    #accuracycalc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
    #print("Testing accuracy: %0.4f"%(accuracycalc))
    saver=tf.train.Saver()
    saver.save(sess, '/deeplearning-food11/1-model')

	
