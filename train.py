
# coding: utf-8

# In[1]:


# %load input_data

import tensorflow as tf
import numpy as np
import os
import pandas as pd
import csv
import numpy as np
from ast import literal_eval
from shutil import copy2
import tarfile


# you need to change this to your data directory
train_dir ='F:/Udacity ML/git/udacity-capstone'
#Function to load the list of image file paths and its labels.
def get_files():
    '''

    Returns:
        list of images and labels

    '''
    #print(tf.__version__)
    print ("Loading files...")
    image_list=[]
    label_list=[]
    trainpath="/valohai/inputs/training-set-images/FinalData_256.tgz"

    train_dir = os.getcwd()
    copy2(trainpath, train_dir)
    tar = tarfile.open(train_dir+"/FinalData_256.tgz")
    tar.extractall(path=train_dir)
    tar.close()

    count=0
    drimage=0
    with open('/valohai/inputs/training-set-labels/ramyaList.l3.csv') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            image_list.append(train_dir+row[0]+".jpeg")
            temp=literal_eval(row[1])
            label_list.append(temp)
            count=count+1
            if temp==1:
                drimage=drimage+1
            if count==6000:
                break
    #print (type(image_list[0]))
    #print (type(label_list[0]))
    print ("DR affected images number ",+drimage)
    print ("Files list loaded.")
    return image_list, label_list

#Function to generate batches of images and labels
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    image = tf.cast(image, tf.string)#cast image list values to string tensors
    label = tf.to_int32(label)

    input_queue = tf.train.slice_input_producer([image, label]) #Produces a slice of each Tensor in [image,label] list.
    #num_epochs: An integer (optional). If specified, slice_input_producer produces each slice num_epochs times
    #before generating an OutOfRange error.
    #If not specified, slice_input_producer can cycle through the slices an unlimited number of times.

    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # data argumentation should go to here ########################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)


    image = tf.image.per_image_standardization(image)
    image=tf.image.random_flip_left_right(image)
    image=tf.image.random_brightness(image, max_delta=0.3)
    image= tf.image.random_contrast(image, 0.8, 1.2)
    image=tf.image.random_flip_up_down(image)
    image=tf.image.random_hue(image,max_delta=0.3)

    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 2,
                                                capacity = capacity)


    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch
x,y=get_files()


# In[2]:


# %load model
import tensorflow as tf

#Function to build the convolution model
def inference(images, batch_size, n_classes,keep_prob):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)


    #conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,16,32],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')


    #pool2 and norm2
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv2, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
    #conv3
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,32,64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')



    #conv4
    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,64,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv3, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name='conv4')


    #pool4 and norm4
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv4, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='SAME', name='pooling1')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm1')
    #conv5
    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,128,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm2, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name='conv5')


    with tf.variable_scope('conv6') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3,3,128,256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv5, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(pre_activation, name='conv6')


    #pool6 and norm6
    with tf.variable_scope('pooling3_lrn') as scope:
        norm3 = tf.nn.lrn(conv6, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75,name='norm3')
        pool3 = tf.nn.max_pool(norm3, ksize=[1,3,3,1], strides=[1,1,1,1],
                               padding='SAME',name='pooling3')

    #local7
    with tf.variable_scope('local7') as scope:
        reshape = tf.reshape(pool3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local7 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        local7=tf.nn.dropout(local7, keep_prob)



    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[1024, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local7, weights), biases, name='softmax_linear')

    return softmax_linear


def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]

    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to
        'sess.run()' call to cause the model to train.

    Args:
        loss: loss tensor, from losses()

    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
      Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """


    with tf.variable_scope('accuracy') as scope:

        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        #roc_score = tf.contrib.metrics.streaming_auc(labels, correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
        #tf.summary.scalar(scope.name+'/roc_score',roc_score)

    #return accuracy,roc_score
    return accuracy



# In[3]:


import os
import numpy as np
import tensorflow as tf



#%%

N_CLASSES = 2
IMG_W = 256  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 256
BATCH_SIZE = 32
CAPACITY = 20
MAX_STEP = 150
learning_rate = 0.01


#%%
def run_training():

    # you need to change the directories to yours.

    logs_train_dir = '/Users/hmakam200/personal/tensorflow/ramya-capstone/board/'

    train, train_label = get_files()
    keep_prob=tf.placeholder(tf.float32,name='keep_prob')

    train_batch, train_label_batch = get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
    train_logits = inference(train_batch, BATCH_SIZE, N_CLASSES,keep_prob=0.7)
    train_loss = losses(train_logits, train_label_batch)
    train_op = trainning(train_loss, learning_rate)
    train__acc =evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    f=open('/Users/hmakam200/personal/tensorflow/ramya-capstone/log.txt', 'w')
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 25 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                f.write("%i %5.2f\n" % (step, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 50 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    f.close()




# In[ ]:


tf.reset_default_graph()
print ("Training started.")

run_training()
print ("training ended.")


# In[5]:


# %load input_data

import tensorflow as tf
import numpy as np
import os
import pandas as pd
import csv
import numpy as np
from ast import literal_eval


# you need to change this to your data directory
train_dir ='F:/Udacity ML/git/udacity-capstone'

def get_files_test():
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels

    '''
    print(tf.__version__)

    image_list=[]
    label_list=[]
    testpath="/Users/hmakam200/Desktop/ramya_data/FinalData_256/"
    with open('test10.csv') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:

            image_list.append(testpath+row[0]+".jpeg")
            temp=literal_eval(row[1])
            label_list.append(temp)
    #print (type(image_list[0]))
    #print (type(label_list[0]))

    return image_list, label_list

def get_batch_test(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)#cast image list values to string tensors
    label = tf.to_int32(label)


    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.per_image_standardization(image)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)





    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 2,
                                                capacity = capacity)



    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


# In[6]:


#%% Evaluate one image
# when training, comment the following codes.


from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def evaluate_test():

    IMG_W = 256
    IMG_H = 256
    BATCH_SIZE = 2
    CAPACITY = 2
    N_CLASSES=2
    tf.reset_default_graph()
    test, test_label = get_files_test()#load the test image paths and labels


    test_batch, test_label_batch = get_batch_test(test,test_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)#get images and lables batches

    coord = tf.train.Coordinator()


    #with tf.Graph().as_default():

    logit = inference(test_batch, BATCH_SIZE, N_CLASSES)
    logit = tf.nn.softmax(logit)
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE,256, 256, 3])

    # you need to change the directories to yours.
    logs_train_dir = '/Users/hmakam200/personal/tensorflow/ramya-capstone/board/'

    saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')

        prediction = sess.run(logit)
        test_acc= sess.run(evaluation(prediction, test_label_batch))
        print (test_acc)
        correct = tf.nn.in_top_k(prediction, test_label_batch, 1)
        #print (tf.shape(correct))
        #print (tf.shape(test_label_batch))
        auc = tf.contrib.metrics.streaming_auc(test_label_batch, correct)
        #sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables()) # try commenting this line and you'll get the error
        roc_score = sess.run(auc)
        print(roc_score)

    coord.request_stop()

    coord.join(threads)
    sess.close()


evaluate_test()
print ("Done")


# In[ ]:
