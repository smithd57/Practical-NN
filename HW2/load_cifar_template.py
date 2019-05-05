import pickle
import numpy as np
import sys


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    

#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):
    """
    Args:
        folder_path: the directory contains data files
        batch_id: training batch id (1,2,3,4,5)
    Return:
        features: numpy array that has shape (10000,3072)
        labels: a list that has length 10000
    """

    ###load batch using pickle###
    file = folder_path + "/data_batch_" + str(batch_id)
    #file = "Alireza" + str(batch_id)
    dictionary = unpickle(file)
    #print(dictionary)
    ###fetch features using the key ['data']###
    features = dictionary[b'data']#dictionary.values()
    ###fetch labels using the key ['labels']###
    labels = dictionary[b'labels']#dictionary.keys()
    #return features,labels
    return dictionary

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
    """
    Args:
        folder_path: the directory contains data files
    Return:
        features: numpy array that has shape (10000,3072)
        labels: a list that has length 10000
    """

    ###load batch using pickle###
    file = folder_path + "/test_batch"
    dictionary = unpickle(file)
    ###fetch features using the key ['data']###
    #features = dictionary[b'data']
    ###fetch labels using the key ['labels']###
    #labels = dictionary[b'labels']
    #return features,labels
    return dictionary

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names(labels):
    label_names = []
    for i in range(len(labels)):
        value = labels[i]
        if(value == 0):
            label_names.append("airplane")
        elif(value == 1):
            label_names.append("automobile")
        elif(value == 2):
            label_names.append("bird")
        elif(value == 3):
            label_names.append("cat")
        elif(value == 4):
            label_names.append("deer")
        elif(value == 5):
            label_names.append("dog")
        elif(value == 6):
            label_names.append("frog")
        elif(value == 7):
            label_names.append("horse")
        elif(value == 8):
            label_names.append("ship")
        else:
            label_names.append("truck")
            
    return label_names

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
    """
    Args:
        features: a numpy array with shape (10000, 3072)
    Return:
        features: a numpy array with shape (10000,32,32,3)
    """
    #loop over the images
    reshaped_img = []
    for i in range(len(features)):
        image = features[i]
        
        #extract red, green, and blue
        red_flat = image[0:1024]
        green_flat = image[1024:2048]
        blue_flat = image[2048:3072]
        
        red_square = np.zeros((32, 32))
        green_square = np.zeros((32, 32))
        blue_square = np.zeros((32, 32))
        
        #convert flat to squares
        for j in range(32):
            #for k in range(32):
            current_range = j*32
            red_square[j] = red_flat[current_range:current_range+32]
            green_square[j] = green_flat[current_range:current_range+32]
            blue_square[j] = blue_flat[current_range:current_range+32]
        
        full_square = np.zeros((32, 32, 3))
        for j in range(32):
            for k in range(32):
                full_square[j][k][0] = red_square[j][k]
                full_square[j][k][1] = green_square[j][k]
                full_square[j][k][2] = blue_square[j][k]
        #group the colors
        #colors = [red_square, green_square, blue_square]
        reshaped_img.append(full_square)
        
    return np.asarray(reshaped_img)

#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id):
    """
    Args:
        folder_path: directory that contains data files
        batch_id: the specific number of batch you want to explore.
        data_id: the specific number of data example you want to visualize
    Return:
        None

    Descrption: 
        1)You can print out the number of images for every class. 
        2)Visualize the image
        3)Print out the minimum and maximum values of pixel 
    """
    pass

#Step 6: define a function that does min-max normalization on input
def normalize(x, size):
    """
    Args:
        x: features, a numpy array
    Return:
        x: normalized features
    """
    #assumes array is in 10000 by 3072 form
    norm = np.zeros((size, 3072))
    
    for i in range(3072):
        #for j in range(10000):
        #find the min and the max of the current col i
        #print(x[:,i])
        #sys.stdout.flush()
        min_val = np.min(x[:,i])
        max_val = np.max(x[:,i])

        norm[:,i] = (x[:,i] - min_val)/(max_val - min_val)
            
    return norm

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x, size):
    """
    Args:
        x: a list of labels
    Return:
        a numpy array that has shape (len(x), # of classes)
    """
    one_hot = np.zeros((size, 10))
    for i in range(size):
        val = x[i]
        one_hot[i][val] = 1
    
    return one_hot

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename, norm_or_not):
    """
    Args:
        features: numpy array
        labels: a list of labels
        filename: the file you want to save the preprocessed data
    """
    norm = normalize(features, len(features))
    #print(norm.shape)
    one_hot = one_hot_encoding(labels, len(labels))
    norm_re = norm
    if norm_or_not: 
        norm_re = features_reshape(norm)
        
    #convert to dictionary
    the_dict = {'data':norm_re, 'labels':one_hot}
    
    #save the file
    with open(filename, 'wb') as handle:
        pickle.dump(the_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #pass
    return

#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path, norm_or_not):
    """
    Args:
        folder_path: the directory contains your data files
    """
    #load training and testing
    train_1 = load_training_batch(folder_path,1)
    features_1 = train_1[b'data']
    labels_1 = train_1[b'labels']
    
    train_2 = load_training_batch(folder_path,2)
    features_2 = train_2[b'data']
    labels_2 = train_2[b'labels']
    
    train_3 = load_training_batch(folder_path,3)
    features_3 = train_3[b'data']
    labels_3 = train_3[b'labels']
    
    train_4 = load_training_batch(folder_path,4)
    features_4 = train_4[b'data']
    labels_4 = train_4[b'labels']
    
    train_5 = load_training_batch(folder_path,5)
    features_5 = train_5[b'data']
    labels_5 = train_5[b'labels']
    
    #split into validation and features
    #features_valid = features_1[9000:10000][:] + features_2[9000:10000][:] + features_3[9000:10000][:] + features_4[9000:10000][:] + features_5[9000:10000][:]
    #labels_valid = labels_1[9000:10000][:] + labels_2[9000:10000][:] + labels_3[9000:10000][:] + labels_4[9000:10000][:] + labels_5[9000:10000][:]
    features_valid = features_5[5000:10000][:]
    labels_valid = labels_5[5000:10000][:]
    
    #features_1 = features_1[0:9000][:]
    #labels_1 = labels_1[0:9000][:]
    
    #features_2 = features_2[0:9000][:]
    #labels_2 = labels_2[0:9000][:]
    
    #features_3 = features_3[0:9000][:]
    #labels_3 = labels_3[0:9000][:]
    
    #features_4 = features_4[0:9000][:]
    #labels_4 = labels_4[0:9000][:]
    
    features_5 = features_5[0:5000][:]
    labels_5 = labels_5[0:5000][:]
    
    test = load_testing_batch(folder_path)
    features_test = test[b'data']
    labels_test = test[b'labels']
    
    #split train_5 to make validation
    
    #process the data
    preprocess_and_save(features_1,labels_1,"train_1", norm_or_not)
    preprocess_and_save(features_2,labels_2,"train_2", norm_or_not)
    preprocess_and_save(features_3,labels_3,"train_3", norm_or_not)
    preprocess_and_save(features_4,labels_4,"train_4", norm_or_not)
    preprocess_and_save(features_5,labels_5,"train_5", norm_or_not)
    
    preprocess_and_save(features_valid,labels_valid,"valid", norm_or_not)
    preprocess_and_save(features_test,labels_test,"test", norm_or_not)
    
    pass

#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
    """
    Args:
        features: features for one batch
        labels: labels for one batch
        mini_batch_size: the mini-batch size you want to use.
    Hint: Use "yield" to generate mini-batch features and labels
    """
    #split the data into batches
    amount_of_data = len(features)
    number_of_bunches = amount_of_data/mini_batch_size
    
    bunches_features = []
    bunches_labels = []
    
    #loop over breaking the data into batches
    for i in range(int(number_of_bunches)):
        current_range = i * mini_batch_size
        f_b = features[current_range:current_range+mini_batch_size]
        l_b = labels[current_range:current_range+mini_batch_size]
        
        bunches_features.append(f_b)
        bunches_labels.append(l_b)
    
    #return the mini-batched data
    return bunches_features, bunches_labels
    

#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
    """
    Args:
        batch_id: the specific training batch you want to load
        mini_batch_size: the number of examples you want to process for one update
    Return:
        mini_batch(features,labels, mini_batch_size)
    """
    file_name = 'train_' + str(batch_id)
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
        
    ###fetch features using the key ['data']###
    features = b['data']
    ###fetch labels using the key ['labels']###
    labels = b['labels']
    #return features,labels
    return mini_batch(features,labels,mini_batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch(mini_batch_size):
    file_name = 'valid'
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
        
    ###fetch features using the key ['data']###
    features = b['data']
    ###fetch labels using the key ['labels']###
    labels = b['labels']
    return mini_batch(features,labels,mini_batch_size)

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    file_name = 'test'
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
        
    ###fetch features using the key ['data']###
    features = b['data']
    ###fetch labels using the key ['labels']###
    labels = b['labels']
    return mini_batch(features,labels,test_mini_batch_size)