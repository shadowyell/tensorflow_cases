
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Softmax-Regression-MNIST" data-toc-modified-id="Softmax-Regression-MNIST-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Softmax Regression MNIST</a></div><div class="lev2 toc-item"><a href="#Input-MNIST-DataSet" data-toc-modified-id="Input-MNIST-DataSet-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Input MNIST DataSet</a></div><div class="lev2 toc-item"><a href="#Define-Data-Structure" data-toc-modified-id="Define-Data-Structure-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Define Data Structure</a></div><div class="lev2 toc-item"><a href="#Forward-Propagation" data-toc-modified-id="Forward-Propagation-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Forward Propagation</a></div><div class="lev2 toc-item"><a href="#Define-The-Loss-Function" data-toc-modified-id="Define-The-Loss-Function-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Define The Loss Function</a></div><div class="lev2 toc-item"><a href="#Define-The-Optimizer-and-Initialize-The-Parameters" data-toc-modified-id="Define-The-Optimizer-and-Initialize-The-Parameters-15"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Define The Optimizer and Initialize The Parameters</a></div><div class="lev2 toc-item"><a href="#Run-The-Procedure" data-toc-modified-id="Run-The-Procedure-16"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Run The Procedure</a></div><div class="lev2 toc-item"><a href="#Test-The-Net" data-toc-modified-id="Test-The-Net-17"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Test The Net</a></div><div class="lev1 toc-item"><a href="#Summary" data-toc-modified-id="Summary-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Summary</a></div>

# # Softmax Regression MNIST
# + The first example from tensorflow book in Tensorflow实战.

# ## Input MNIST DataSet

# In[13]:

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


# In[14]:

#from tensorflow.examples.tutorials.mnist import input_data for 1.3.0 version
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[15]:

# train, validation, test
# imagesize 28*28
print (mnist.train.images.shape, mnist.train.labels.shape)
print (mnist.test.images.shape, mnist.test.labels.shape)
print (mnist.validation.images.shape, mnist.validation.labels.shape)


# ## Define Data Structure
# + Interactive Session - Run codes in here, Independent zone for data and computation
# + Placeholder - Place to input the data
# + Define the weights and biases

# In[16]:

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# ## Forward Propagation

# In[17]:

y = tf.nn.softmax(tf.matmul(x, W) + b)


# ## Define The Loss Function
# + Placeholder - Place the true label
# + Define Cross-Entropy NODE= $-\sum_i{y_{true}\log(y_{pred})}$

# In[18]:

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# ## Define The Optimizer and Initialize The Parameters
# + Use GD and Define train_step NODE
# + Use Global-variables-initializer

# In[19]:

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()


# ## Run The Procedure
# + Batch Size = 100

# In[55]:

for epoch in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})


# ## Test The Net
# + Define the computation graph and NODE accuracy
# + then use eval() to implement the procedure

# In[56]:

correct_prediction = tf.equal(tf.argmax(y ,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[57]:

print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))


# # Summary
# 1. Define the propagation structure(computation graph)
# 2. Define the loss function and the optimizer
# 3. Training 
# 4. Testing
