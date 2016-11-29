import tensorflow as tf
import numpy as np

#######################################################

def inference(x):
    # Weight vector.
    W = tf.Variable(tf.zeros([1,1]))
    # Bias vector.
    b = tf.Variable(tf.zeros([1]))
    # Multiplication using matmul.
    y = tf.matmul(x, W) + b
    return y

#######################################################

def loss(y, y_):
    # We square the difference of all the values of layers.
    # Equvalent for j(theta)
    cost = tf.reduce_sum(tf.pow((y_ - y),2))
    return cost

#######################################################

def training(cost):
    # Using tensorflow built in gradient decent optimizer.
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
    return train_step   

#######################################################

def evaluate(y, y_):
    #a = tf.argmax(y,1)
    #b = tf.argmax(y_,1)
    correct_prediction = y #tf.equal(y, y_)
    # Converting the value of y to float32.
    float_val = tf.cast(correct_prediction,tf.float32)

    prediction_as_float = tf.reduce_mean(float_val)
    return prediction_as_float

#######################################################
x = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

#W = tf.Variable(tf.zeros([1,1]))
#b = tf.Variable(tf.zeros([1]))
#y = tf.matmul(x, W) + b

# Execution Steps.
y = inference(x)
# Cost function return calculated loss per w.
cost = loss(y, y_)
# Gradient Decent optiizer.
train_step = training(cost)
# Finally Evealuate.
eval_op = evaluate(y, y_)

#cost = tf.reduce_sum(tf.pow((y_ - y),2))
#train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

###########################################
# Session and initgialization.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

###########################################
steps = 100
for i in range(steps):
    xs = np.array([[i]])   #house size are the index
    ys = np.array([[5*i]]) #house price 5 time the index
    # This makes 100 samples of data.
    # feed = {internal:house_size, y_true:House_prices}
    feed = {x:xs, y_:ys}
    # Run 
    sess.run(train_step, feed_dict=feed)    
    
    print("After %d iteration: " % i)
    #print("W: %f" % sess.run(W))
    #print("b: %f" % sess.run(b))
    ##########################################

for i in range(100,200):
    xs_test = np.array([[i]])   #house size
    ys_test = np.array([[2*i]]) #house price
    # A dictionary.
    feed_test = {x:xs_test, y_:ys_test}
    # Evaluate results in returning
    result = sess.run(eval_op, feed_dict=feed_test)
    #print sess.run(y)
    print "Run {},{}".format(i, result)
    x_input = raw_input()

