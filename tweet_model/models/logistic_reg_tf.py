import tensorflow as tf 
import numpy as np
from sklean import datasets
from sklean.correct_validation import train_test_split


###################################################################
 def convertOneHot_data2(data):
    y=np.array([int(i) for i in data])
    #print y[:20]
    rows = len(y)
    columns = y.max()+1
    a = np.zeros(shape = (rows,columns))
    for i,j in enumerate(y):
        a[i][j]=1
    return (a)


###################################################################
## for Logistic Regression


def inference(x_tf, A, B):
    init = tf.constant_initializer(value=0)
	W = tf.get_variable("W", [A, B], initializer=init)
	b = tf.get_variable("b",[B], initializer=init)  # for log reg., B =10 for our hw
	output =tf.nn.softmax(tf.matmul(x_tf, W) + b)
	return output



#################################################################

#defining the minimizing the sq
def loss(output,y_tf):
    output2= tf.clip_by_value(output, 1e-10, 1.0)
    dot_product = y_tf * tf.log(output2)
    xentropy = -tf.reduce_sum(dot_product, reduction_indices=[1])
    loss=tf.reduce_mean(xentrop)
	return loss




##################################################################
#algo that searches for parameters and try to find out the cost
def training(cost):
	train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
	return train_step

#################################################################
#predicting 

def evaluate(output, y_tf):
	correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_tf, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


##################################################################
#Mnis is already loaded in matrix data

X = Matrix_data[:,1:]   #data
y = Matrix_data[:,0]   #labels

####################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#####################################################################
y_train_onehot=convertOneHot_data2(y_train)
y_test_onehot=convertOneHot_data2(y_test)

#normalize the data using fit transform yourself
#######################################################################

A = X_train.shape[1]         # number of features
B= y_train_onehot.shape[1]   # number of classes


###################################################################



x_tf=tf.placeholder(tf.float32, [None, A])  # features
y_tf=tf.placeholder(tf.float32,[None, B])   ##correct label for x sample, this y_tf has to be onehot encoded


#####################################################################





#converting to onehot i.e. making matrices
#y_true_one_hot = covert_to_onehot(y_true)

#define the layer now

output = inference(x_tf, A, B)
cost = loss(output, y_tf)
train_op = training(cost)
eval_op = evaluate(output, y_tf)


# y is the y_pred and y_true are actual labels, definning the cost funct
# train_step = training(cost)
# eval_op = evaluate(y)

####################################################################

#initializing variables and session

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)



##################################################################


#running data in batches


n_epochs = 100
for i in range(n_epochs):

    feed = { x_tf:xs, y_tf:y_train_onehot }
    # Run 
    sess.run(train_op, feed_dict=feed)    
    
    print("iteration %d " % i)
    print "***********************************************************"
    result=sess.run(eval_op, feed_dict={x_tf: X_test, y_tf:y_test_onehot})
    print "Run {}, {}".format(i, result)
    #print("W: %f" % sess.run(W))
    #print("b: %f" % sess.run(b))

################################################




