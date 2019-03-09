import numpy as np
import scipy.io as sio
import tensorflow as tf


def one_hot_vector(i):
    one_hot = # Construct a 1D matrix whose cells are filled by zero
    one_hot[i] = # Reassign the position i of one_hot to one
    return one_hot


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

target_vector_one_hot = np.zeros([number_of_samples, number_of_labels])
for i in range(number_of_samples):
    target_vector_one_hot[i, :] = one_hot_vector(target_vector[i] % 10)

'''
Step 1: Construct TF graph
'''

X = tf.placeholder(tf.float32, [None, number_of_features], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')  # 0-9 digits recognition. Hence, 10 classes !

thetas = tf.Variable(tf.zeros([number_of_features, 10]), name='Thetas')
theta0 = tf.Variable(tf.zeros([10]), name='theta0')

'''
Step 2: Define the hypothesis function i.e. Softmax function (see Slide# )
'''

hypothesis_function = # Put your code here 

'''
Step 3: Define the cost function i.e. the cross-entropy loss
'''

cost_function = # Put your code here

'''
Step 4: Use gradient descent with learning rate of 0.0003 to minimize the cost function
'''

optimizer = # Put your code here 

'''
Step 5: Now, we prepare examples for visualizing our test results
'''

# We sample 10 images from the training dataset
random_sample_index = np.random.choice(feature_matrix.shape[0], 10)
feature_random_sample = feature_matrix[random_sample_index, :]
actual_random_sample = target_vector[random_sample_index].T

# Display each randomized image
plt.imshow(feature_random_sample.reshape(-1, 20).T)
plt.axis('off')

class_probability_sample = np.zeros(shape=[10, number_of_labels])

'''
Step 6: Train the model 
'''

with tf.Session() as session:
    # Put your code here to: 
    # 1) train the model
    # 2) compute the cost for each training step
    # 3) report the number of correct prediction and its accuracy
    # Noted that there may be more than one line of code for this step !

    # Class probability of 10 sampled images
    class_probability_sample = session.run(hypothesis_function, feed_dict={X: feature_random_sample})

    # Compare the actual digits against predicted digits
    print("Actual digit:\n {}".format(actual_random_sample % 10))
    print("Predicted digit:\n {}".format(class_probability_sample.argmax(axis=1)))

