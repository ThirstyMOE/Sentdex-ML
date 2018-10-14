import tensorflow as tf
from create_sentiment_featureset import create_feature_sets_and_labels
import numpy as np

# TODO: load in from pickle
# Data here needs to be massive for language processing
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

# Define layers and node amounts per layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
# Define number of input dimensions
n_input_dimension = len(train_x[0])
# Define number of output classes
n_classes = 2
# Hundred samples of data per batch
batch_size = 100

# placeholder tensorflow float vectors. x is input (R 784), y is float output
x = tf.placeholder('float', [None, n_input_dimension])
y = tf.placeholder('float')

'''
    pre: takes data vector in dimensions n_input_dimension,
        if not in correct shape, tf throws error
    post: returns the output layer in shape of n_classes, with it part of the tf
        computation graph
'''
def neural_network_model(data):
    # Define hidden layers and output layer, all weights and biases are initialized to
    # random values in the shape of each prescribed layer.
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_input_dimension, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    # Vector add for each layer's matrix multiplied previous layer to current layer
    # Then apply activation function (relu) on each layer
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    # Get prediction layer outputs from tf computation graph
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    # Define cost function
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # Define optimization algorithm
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Define how many epochs to go across data for
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                # Take a batch of data
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

                i += batch_size

            print('Epoch', epoch + 1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)
