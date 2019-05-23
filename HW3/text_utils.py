import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

"""
Implement a class object that should have the following functions:
1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and 
validation mini batches.
2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
        b1)create self.char that is a tuple contains all unique character appeared in the txt input.
        b2)create self.vocab_size that is the number of unique character appeared in the txt input.
        b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    d)map all characters of training and validation data to their integer label and save as 'npy' files respectively.
3)helper function to load preprocessed data
4)helper functions to create training and validation mini batches
"""
class TextLoader():
    def __init__(self, directory, batch_size, seq_len):
        #define the internal variables
        self.directory = directory
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.char = None
        self.vocab_size = None
        self.vocab = {}
        self.vocab_reverse = {}
        
        #get the text and then split it
        text = (self.read_data()[0:100000]).lower()
        #print(len(text))
        self.split = [i for i in text]
        #print(len(self.split))
        
        #split = text.list()
        
        #create the arrays
        self.txt_parameters()
        
        #create the training and valid sets
        text_train_in, text_valid_in, text_train_out, text_valid_out = self.split_data()
        #print(text_train_in.shape)
        
        #save the data
        self.save_data(text_train_in, text_valid_in, text_train_out, text_valid_out)
        
        #return text_train_in, text_valid_in, text_train_out, text_valid_out
    
    def read_data(self):
        with codecs.open(self.directory, 'r', encoding='utf8') as f:
            text = f.read()
        return text
        
    def txt_parameters(self):
        #find the unique arrays and their counts
        self.char, self.vocab_size = np.unique(self.split, return_counts =True)
        print(self.char)
        #print(self.vocab_size)
        
        #fill the dictionary
        for i in range(len(self.char)):
            self.vocab[self.char[i]] = i
            self.vocab_reverse[i] = self.char[i]
        
        
    def split_data(self):
        
        #convert the data from text to numbers
        text_as_ints = []
        #setup a one_hot encoder
        for i in range(len(self.split)):
            curr_char = self.split[i]
            matching_int = self.vocab[curr_char]
            #one_hot = np.zeros((len(self.char)))
            #one_hot[matching_int] = 1
            text_as_ints.append([matching_int])
        
        
        #rearrange the data into a shape of [data_num, num_seq, features]
        the_range = len(text_as_ints) - self.seq_len
        #print("range")
        #print(the_range)
        text_sets = np.zeros([the_range, self.seq_len+1, 1])
        for i in range(the_range):
            #split out the sequence
            chunk = text_as_ints[i:i+self.seq_len+1]
            text_sets[i] = chunk

        #print(text_sets[0])

        #shuffle on first axis
        np.random.shuffle(text_sets)

        #split into input and output
        text_in = np.zeros([the_range, self.seq_len, 1])
        text_out = np.zeros([the_range, len(self.char)])

        for i in range(len(text_sets)):
            the_in = text_sets[i][0:self.seq_len]
            the_out = text_sets[i][self.seq_len]
            #print(the_out)
            #print(the_out.shape)
            #print(the_out[0])
            
            one_hot = np.zeros((len(self.char)))
            one_hot[int(the_out[0])] = 1

            text_in[i] = the_in/len(self.char)
            text_out[i] = one_hot

        #print("text_in", text_in[0])
        #print("text_out", text_out[0])
        #break things into batches
        batch_length = int(len(text_in)/self.batch_size)
        #print("batch_length", batch_length)
        text_in_batched = np.zeros([batch_length, self.batch_size, self.seq_len, 1])
        text_out_batched = np.zeros([batch_length, self.batch_size, len(self.char)])

        for i in range(batch_length):
            step = i*self.batch_size

            chunk = text_in[step:step+self.batch_size]
            text_in_batched[i] = chunk

            chunk = text_out[step:step+self.batch_size]
            text_out_batched[i] = chunk


        #redefine range
        the_range = len(text_in_batched)
        #print("the_range", the_range)

        #split into training, validation, and testing by a 90%, 10%
        train_range = int(the_range*0.9)
        #print("train_range", train_range)
        #valid_range = int(the_range*0.05) + train_range
        #test_range = the_range - train_range-valid_range

        #tesla split
        text_train_in = text_in_batched[0:train_range+1]
        text_valid_in = text_in_batched[train_range:the_range]

        text_train_out = text_out_batched[0:train_range+1]
        text_valid_out = text_out_batched[train_range:the_range]
        
        return text_train_in, text_valid_in, text_train_out, text_valid_out
        
    def save_data(self, text_train_in, text_valid_in, text_train_out, text_valid_out):
        #save self.char
        with open("uni_char", 'wb') as handle:
            cPickle.dump(self.vocab, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        
        #save the arrays
        np.save("text_train_in", text_train_in)
        np.save("text_train_out", text_train_out)
        np.save("text_valid_in", text_valid_in)
        np.save("text_valid_out", text_valid_out)
        
    def load_data(self):
        text_train_in = np.load("text_train_in.npy")
        text_train_out = np.load("text_train_out.npy")
        text_valid_in = np.load("text_valid_in.npy")
        text_valid_out = np.load("text_valid_out.npy")
        
        return text_train_in, text_valid_in, text_train_out, text_valid_out
        