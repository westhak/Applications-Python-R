"""
Data: MNIST Database
Task: Classification of handwritten digits from MNIST database using single layer neural network with logistic regression
      Compare Classification Accuracy of GradientDescent and Adam Optimizer while varying - iterations and batchsize
Packages: numpy, tensorflow, matplotlib
@author: Swetha
"""



import gzip
import hashlib
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from itertools import cycle

#Enter your id here
id = "xxxx"

your_random_seed =  int(hashlib.md5(id.encode()).hexdigest(),16)% 15485863
np.random.seed(your_random_seed)
tf.set_random_seed(your_random_seed)

#Open the data

with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    train_images_raw = f.read()
    
with gzip.open('train-labels-idx1-ubyte.gz','rb') as f:
    train_labels_raw = f.read()

with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images_raw = f.read()

with gzip.open('t10k-labels-idx1-ubyte.gz','rb') as f:
    test_labels_raw = f.read()

#Get Images from raw data file
    
train_labels=np.array(bytearray(train_labels_raw[8:])).astype('int32')
test_labels=np.array(bytearray(test_labels_raw[8:])).astype('int32')


numI=60000
dima=28
dimb=28
train_images=np.array(bytearray(train_images_raw[16:]))
train_images=train_images.reshape(60000,28*28) 

test_images=np.array(bytearray(test_images_raw[16:]))
test_images=test_images.reshape(10000,28*28) 

train_images.shape,test_images.shape, train_labels.shape, test_labels.shape
#image_one = train_images[0,:]

#PLot the first train image
image_one=train_images[0,:]
plt.imshow(image_one.reshape(28,28),cmap=plt.cm.gray)
plt.show()



#Split the data into batches
batch_size = 100
assert train_images.shape[0] % batch_size == 0
number_of_batches = int(train_images.shape[0] / batch_size)

def generate_batches(list_of_arrays, number_of_batches):
    batched_train_labels = []
    batched_train_images = []

    for i in range(0,number_of_batches):
        batch0 = list_of_arrays[0][(i*batch_size) : (i +1)* batch_size]
        batch1 = list_of_arrays[1][(i*batch_size) : (i +1)* batch_size]
        batched_train_images.append(batch0)
        batched_train_labels.append(batch1)
    batched = [batched_train_images,batched_train_labels]
    return batched

list_of_arrays = [train_images, train_labels]
batched_train_images, batched_train_labels = generate_batches([train_images, train_labels], number_of_batches)
print(batched_train_labels[0:5])

# CLASSIFICATION FUNCTIONS

#Classification function returning accuracy score
def classification_with_given_optimizer(optimizer):
        # Parameters
        total_number_of_epochs = 20
        total_number_of_iterations = total_number_of_epochs * number_of_batches
        accuracy_epoch = np.zeros(total_number_of_epochs)
        
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
        # Set model weights
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        # Construct model
        output = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
        # Minimize error using cross entropy
        loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
        
        # Gradient Descent or AdamOptimizer

        train_op = optimizer(learning_rate=0.05).minimize(loss)
        
                               
        # Initialize the variables 
        sess = tf.InteractiveSession() # this opens the session
        tf.global_variables_initializer().run() # this initializes the variables

        # Training cycle
        for epoch in range(total_number_of_epochs):
            avg_cost = 0
            # Loop over all batches
            for i in range(number_of_batches):
                batch_xs= batched_train_images[i]      
                batch_ys0 =batched_train_labels[i]
                #Convert batch_ys into onehotvector
                batch_ys = np.zeros((100,10))
                batch_ys[np.arange(100), batch_ys0] = 1

                # Run optimization op (backprop) and cost op (to get loss value)
                _,c,pred = sess.run([train_op,loss,output], feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost = avg_cost + c / number_of_batches

            #print("Epoch", '%01d' % (epoch+1),'\n',"Avg_cost:", "{:.5f}".format(avg_cost))

            # Test model
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            test_labels_hot =   np.zeros((10000,10))
            test_labels_hot[np.arange(10000), test_labels] = 1
            accuracy_epoch[epoch] =accuracy.eval({x: test_images, y: test_labels_hot}) 
            #print("Accuracy:", accuracy_epoch[epoch] ,'\n')
        return accuracy_epoch

#Run the classification function
accuracy_gradient_descent = classification_with_given_optimizer(tf.train.GradientDescentOptimizer)
accuracy_adam = classification_with_given_optimizer(tf.train.AdamOptimizer)

#Plot the results
plt.plot(accuracy_gradient_descent, "r", label="GD")
plt.plot(accuracy_adam, "b", label="Adam")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

#Observe changes in accuracy with number of iterations

def classification_with_given_optimizer_with_total_number_of_iteration(optimizer, total_number_of_iterations=50):
        # Parameters
        total_number_of_iterations = 50
        accuracy_iterations = np.zeros(total_number_of_iterations)
        
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
        # Set model weights
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        # Construct model
        output = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
        # Minimize error using cross entropy
        loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
        # Gradient Descent
        train_op = optimizer(0.05).minimize(loss)
       
        # Initialize the variables 
        sess = tf.InteractiveSession() # this opens the session
        tf.global_variables_initializer().run() # this initializes the variables

        # Training cycle
        for i in range(total_number_of_iterations):
            avg_cost = 0
            # Loop over all batches
            #for i in range(number_of_batches):
            xs= train_images      
            ys0 =train_labels
                #Convert batch_ys into onehotvector
            ys = np.zeros((60000,10))
            ys[np.arange(60000), ys0] = 1

            # Run optimization op (backprop) and cost op (to get loss value)
            _,c,pred = sess.run([train_op,loss,output], feed_dict={x: xs, y: ys})
            # Compute average loss
            avg_cost = avg_cost + c / number_of_batches

            # Test model
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            test_labels_hot =   np.zeros((10000,10))
            test_labels_hot[np.arange(10000), test_labels] = 1
            accuracy_iterations[i] =accuracy.eval({x: test_images, y: test_labels_hot}) 
        #    print("Accuracy:", accuracy_iterations[i] ,'\n')
        return accuracy_iterations

#Run the classification function
accuracy_gradient_descent_by_iteration = classification_with_given_optimizer_with_total_number_of_iteration(tf.train.GradientDescentOptimizer)
accuracy_adam_by_iteration = classification_with_given_optimizer_with_total_number_of_iteration(tf.train.AdamOptimizer)

#Plot the results
plt.plot(accuracy_gradient_descent_by_iteration, "r", label="GD")
plt.plot(accuracy_adam_by_iteration, "b", label="Adam")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()

#Classification function returning accuracy with varying batch size

def classification_with_given_optimizer_batch_size(optimizer, batch_size, total_number_of_epochs):
    number_of_batches =int(train_images.shape[0]/batch_size)
    batched_train_labels = []
    batched_train_images = []
#    print(number_of_batches)
    for i in range(0,number_of_batches):
        batch0 = list_of_arrays[0][(i*batch_size) : (i +1)* batch_size]
        batch1 = list_of_arrays[1][(i*batch_size) : (i +1)* batch_size]
        batched_train_images.append(batch0)
        batched_train_labels.append(batch1)
    batched = [batched_train_images,batched_train_labels]
    
    total_number_of_iterations = total_number_of_epochs * number_of_batches
    accuracy_final = 0
    
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # Construct model
    output = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    # Minimize error using cross entropy
    loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

    # Gradient Descent or AdamOptimizer

    train_op = optimizer(learning_rate=0.05).minimize(loss)


    # Initialize the variables 
    sess = tf.InteractiveSession() # this opens the session
    tf.global_variables_initializer().run() # this initializes the variables

    # Training cycle
    for epoch in range(total_number_of_epochs):
        avg_cost = 0
        # Loop over all batches
        for i in range(number_of_batches):
            batch_xs= batched_train_images[i]      
            batch_ys0 =batched_train_labels[i]
            #Convert batch_ys into onehotvector
            batch_ys = np.zeros((batch_size,10))
            batch_ys[np.arange(batch_size), batch_ys0] = 1

            # Run optimization op (backprop) and cost op (to get loss value)
            _,c,pred = sess.run([train_op,loss,output], feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost = avg_cost + c / number_of_batches

        # Test model
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_labels_hot =   np.zeros((10000,10))
    test_labels_hot[np.arange(10000), test_labels] = 1
    accuracy_final =accuracy.eval({x: test_images, y: test_labels_hot}) 
#    print(accuracy_final)
    return accuracy_final


accuracy_gradient_descent_by_batch_size=np.zeros(5)
accuracy_adam_by_batch_size = np.zeros(5)
b=np.array([100,600,1000,6000,10000])
for i in range(5):
    batch_size = b[i]
    accuracy_gradient_descent_by_batch_size[i] = classification_with_given_optimizer_batch_size(tf.train.GradientDescentOptimizer,batch_size,total_number_of_epochs=20)
    accuracy_adam_by_batch_size[i] = classification_with_given_optimizer_batch_size(tf.train.AdamOptimizer,batch_size,total_number_of_epochs=20)

x=b
y1=accuracy_gradient_descent_by_batch_size
y2=accuracy_adam_by_batch_size

#Plot the results
plt.plot(x,y1, "r", label="GD")
plt.plot(x,y2, "b", label="Adam")
plt.xlabel('Batch_Size')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()


    