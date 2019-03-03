import numpy as np
import scipy.io as sio
import tensorflow as tf


'''
Step 0: Load the input data, which is stored in digits.mat file. 
Then, initialize some variables namely number_of_samples, number_of_features, and number_of_labels
'''

data = sio.loadmat("data/digits.mat")

feature_matrix = data['X']  # the feature matrix is labeled with 'X' inside the file
target_vector = np.squeeze(data['y'])  # the target variable vector is labeled with 'y' inside the file

number_of_samples = feature_matrix.shape[0]  # i.e. 5000 samples
number_of_features = feature_matrix.shape[1]  # i.e. 400 features (20 * 20)
number_of_labels = np.max(target_vector)  # i.e. 1, 2, 3, ..., 10

'''
Step 1: Construct TF graph
'''

X = tf.placeholder(tf.float32, [None, number_of_features], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

thetas = tf.Variable(tf.zeros([number_of_features, 1]), name='Thetas')
theta0 = tf.Variable(tf.zeros([1]), name='theta0')

'''
Step 2: Define the hypothesis function
'''

z = tf.matmul(X, thetas)
hypothesis_function = # wrute the hypothesis function based on TF methods here 

'''
Step 3: Define the cost function
'''

cost_function = # write the cost function based on TF methods here

'''
Step 4: Use gradient descent with learning rate of 0.6 to minimize the cost function
'''

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.6).minimize(cost_function)

'''
Step 5: Now, we are ready to train our classifiers. Since we are implementing the one-vs-all method, 
then we need 10 classifiers i.e. we need to create a 10x401 array for storing thetas, where: 
    - each row describes parameters of a particular classifier and 
    - each column indicates a particular feature.  
    
Also, we need to create another array for storing probabilities of class memberships. The size of this array 
should be 5000x10, where :
    - rows indicates the dataset and 
    - columns indicates classes.
'''

classifier = # Based on the above explanation, what code should we put here?  
class_probabilities = # Based on the above explanation, what code should we put here? 

with tf.Session() as session:
    '''
    Step 6: Initialize the necessary variables
    '''
    session.run(tf.global_variables_initializer())

    for i in range(number_of_labels):
        print("Running the {}th-classifier".format(i+1))
        label = (target_vector == (i + 1)).astype(int)

        for j in range(5000):
            _, cost = # run the optimizer and cost function in a session w.r.t. matrices Input and Output})

        print("Cost = {}\n".format(cost))

        # classifier[0, :] corresponds to digit 1 whereas classifier[9, :] corresponds to digit 10.
        classifier[i, :] = np.concatenate([np.matrix(session.run(theta0)), np.matrix(session.run(thetas)).T], axis=1)
        class_probabilities[:, i] = # run the hypothesis function to calculate the class probability for this particular classifier

    predictions = class_probabilities.argmax(axis=1)


print("Training accuracy: {} %".format(100 * np.mean((predictions + 1) == target_vector)))
