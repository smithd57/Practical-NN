import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.contrib.layers import flatten

"""
TO: Define your char rnn model here

You will define two functions inside the class object:

1) __init__(self, args_1, args_2, ... ,args_n):

    The initialization function receives all hyperparameters as arguments.

    Some necessary arguments will be: batch size, sequence_length, vocabulary size (number of unique characters), rnn size,
    number of layers, whether use dropout, learning rate, use embedding or one hot encoding,
    and whether in training or testing,etc.

    You will also define the tensorflow operations here. (placeholder, rnn model, loss function, training operation, etc.)


2) sample(self, sess, char, vocab, n, starting_string):
    
    Once you finish training, you will use this function to generate new text

    args:
        sess: tensorflow session
        char: a tuple that contains all unique characters appeared in the text data
        vocab: the dictionary that contains the pair of unique character and its assoicated integer label.
        n: a integer that indicates how many characters you want to generate
        starting string: a string that is the initial part of your new text. ex: 'The '

    return:
        a string that contains the genereated text

"""
class Model():
    def __init__(self, seq_len, rnn_size, num_layers, learning_rate, vocab_size):
        self.seq_len = seq_len
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.vocab_size = vocab_size
        
        tf.reset_default_graph()
        #tf graph input
        self.X = tf.placeholder(tf.float32,[None,self.seq_len,1],name='X')
        self.Y = tf.placeholder(tf.float32,[None,self.vocab_size],name='Y')

        def RNN(x):   
            # create a BasicRNNCell
            rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.rnn_size), 1, 0.8) for i in range(self.num_layers)]

            # create a RNN cell composed sequentially of a number of RNNCells
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
            #rnn_cell = tf.nn.rnn_cell.LTSMCell(self.rnn_size)

            # 'state' is a tensor of shape [batch_size, cell_state_size]
            outputs, state = tf.nn.dynamic_rnn(multi_rnn_cell, x, dtype=tf.float32)#,initial_state=initial_state)

            #flatten to connect to fully connected
            #concat = tf.concat(outputs, 1)
            #full_in = tf.reshape(concat, [-1, self.rnn_size])
            full_in = flatten(outputs[:,seq_len-1,:])

            #fully connected layer
            full1 = tf.contrib.layers.fully_connected(inputs=full_in, num_outputs=self.vocab_size, activation_fn=None) #num outputs 67 if one_hot

            return full1 #has shape [batch_size*seq_len, 1 or 67(OH)]
        #predicted labels
        self.logits = RNN(self.X)

        #define loss
        #just_soft = tf.nn.softmax(logits=logits)
        #y_0 = tf.shape(self.Y)[0]#shape
        #y_1 = tf.shape(self.Y)[1]

        #print(tf.shape(features))
        #y_reshape = tf.reshape(self.Y, [y_0*y_1, self.num_char])
        #self.loss = tf.reduce_mean(tf.math.abs(self.logits - y_reshape),name='loss')
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,labels=self.Y),name='loss')
        #define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def sample(self, sess, vocab, vocab_reverse, n, start):
        #loop over the string
        #self.sess = sess
        #self.vocab = vocab
        #self.n = n
        #self.start = start
        
        #break up the string
        split = [i for i in start]
        
        #setup a one_hot encoder
        for i in range(len(split)):
            #print(type(split[0]))
            val = split[i]
            location = vocab[val]
            #one_hot = np.zeros((self.num_char))
            #one_hot[location] = 1
            split[i] = [location/self.vocab_size] 
        
        for i in range(n):
            length = len(split)
            
            #grab the sequence length
            piece = split[length - self.seq_len : length]
            
            batch_x = [piece]
            #batch_y = tesla_test_out[0]
            #run optimization
            #print(batch_x)
            #print(batch_x)
            next_chars = sess.run(self.logits, feed_dict={self.X:batch_x})
            val = np.argmax(next_chars)
            #print(val)
            split.append([val/self.vocab_size])
            #print(split)
            #print(split)
            #test_range.append(i)
            #correct_guess.append(preserve_tesla[i][5])
            #net_guess.append(guesses[0])
        
        #print(split)
        #decode
        string_back = []
        for i in range(len(split)):
            val = split[i][0] * self.vocab_size
            string_back.append(vocab_reverse[val])
            
        
        return string_back