import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def preprocess():
    """
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    # Adding bias term to the input data
    bias_term = np.ones((n_data, 1))
    train_data = np.column_stack((bias_term, train_data))

    # Reshaping th weights
    weights = initialWeights.reshape((n_features + 1, 1))

    # Applying sigmoid function to the probabilities
    z = np.dot(train_data, weights)
    theta_z = sigmoid(z)

    # Calculating cross-entropy error for negative log-likelihood
    e1 = labeli * np.log(theta_z)
    e2 = (1 - labeli) * np.log(1 - theta_z)
    error = -np.mean(e1 + e2)

    # Calculating gradient error with repective to the weights
    error_grad = (1.0 / n_data)*np.dot(np.transpose(train_data), (theta_z - labeli))
    error_grad = error_grad.flatten()

    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    n_data = data.shape[0]

    # Adding bias term to input data
    bias_term = np.ones((n_data, 1))
    data_with_bias = np.column_stack((bias_term, data))  # Now data has shape (N, D+1)

    # Calculating class probabilities for all 10 classifiers classes
    z = np.dot(data_with_bias, W)
    theta = sigmoid(z)

    # Getting the class with the higher probability corresponding to the predicted class
    class_indices = np.argmax(theta, axis=1)

    # Reshaping it result to column vector
    label = class_indices.reshape(-1, 1)

    return label

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Adding bias term to train_data
    bias_term = np.ones((n_data, 1))
    train_data_with_bias = np.hstack((bias_term, train_data))  # Shape becomes (N, D+1)

    # Reshape weights to matrix form (D+1) x 10
    weights = params.reshape((n_feature + 1, n_class))

    # Compute logits (z)
    logits = np.dot(train_data_with_bias, weights)

    # Appyling softmax function to convert the logits to probabilities
    exp_logits = np.exp(logits)
    softmax_denominator = np.sum(exp_logits, axis=1, keepdims=True)
    theta = exp_logits / softmax_denominator

    # Calculating cross-entropy error to measure the difference
    log_likelihood = labeli * np.log(theta)
    error = -np.sum(log_likelihood) / n_data

    # Computing error gradient as the derivative of the cross-entropy loss
    gradient = np.dot(np.transpose(train_data_with_bias), (theta - labeli)) / n_data
    error_grad = gradient.flatten()  # Flatten for compatibility with minimize

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    # Adding bias term to data
    n_data = data.shape[0]
    bias_term = np.ones((n_data, 1))
    data_with_bias = np.hstack((bias_term, data))

    # Compute logits (z)
    logits = np.dot(data_with_bias, W)  # Shape (N, 10)

    # Apply the softmax function to compute probabilities
    exp_logits = np.exp(logits)
    softmax_denominator = np.sum(exp_logits, axis=1, keepdims=True)
    softmax_probabilities = exp_logits / softmax_denominator

    # Assigning each input with the highest probability for each class
    predicted_labels = np.argmax(softmax_probabilities, axis=1).reshape(-1, 1)

    # Storing the predicted labels
    label[:] = predicted_labels

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights.flatten(), jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################
## SVM with linear kernel ##
print('SVM using linear kernel')
svm = SVC(kernel='linear')
svm.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100*svm.score(train_data, train_label)) )
print('\n Validation set Accuracy:' + str(100*svm.score(validation_data, validation_label)) )
print('\n Testing set Accuracy:' + str(100*svm.score(test_data, test_label)) )

## SVM with radial basis function, gamma = 1 ##
print('SVM using rbf with value of gamma setting to 1')
svm_rbf = SVC(kernel='rbf', gamma=1)  
svm_rbf.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * svm_rbf.score(train_data, train_label)))
print('\n Validation set Accuracy:' + str(100 * svm_rbf.score(validation_data, validation_label)))
print('\n Testing set Accuracy:' + str(100 * svm_rbf.score(test_data, test_label)))

## SVM using rbf with value of gamma setting to default
print('SVM using rbf with value of gamma setting to default')
svm_rbf_def = SVC(kernel='rbf')
svm_rbf_def.fit(train_data, train_label.flatten())
print('\n Training set Accuracy:' + str(100 * svm_rbf_def.score(train_data, train_label)))
print('\n Validation set Accuracy:' + str(100 * svm_rbf_def.score(validation_data, validation_label)))
print('\n Testing set Accuracy:' + str(100 * svm_rbf_def.score(test_data, test_label)))


##SVM using rbf with value of gamma setting to default and varying value of C 
print('SVM using rbf with value of gamma setting to default and varying value of C ')
accuracy_results = {'C': [], 'Train': [], 'Validation': [], 'Test': []}

print(f"{'C':<10}{'Train Acc (%)':<15}{'Val Acc (%)':<15}{'Test Acc (%)':<15}")
print("-" * 50)

# Iterating through different C values
for C in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    svm_rbf = SVC(kernel='rbf', C=C)
    svm_rbf.fit(train_data, train_label.ravel())

    train_acc = accuracy_score(train_label, svm_rbf.predict(train_data)) * 100
    val_acc = accuracy_score(validation_label, svm_rbf.predict(validation_data)) * 100
    test_acc = accuracy_score(test_label, svm_rbf.predict(test_data)) * 100

    # Add results to the dictionary
    accuracy_results['C'].append(C)
    accuracy_results['Train'].append(train_acc)
    accuracy_results['Validation'].append(val_acc)
    accuracy_results['Test'].append(test_acc)

    print(f"{C:<10}{train_acc:<15.2f}{val_acc:<15.2f}{test_acc:<15.2f}")

# Plot the accuracies against C values
plt.figure(figsize=(10, 6))
plt.plot(accuracy_results['C'], accuracy_results['Train'], marker='o', label='Training Accuracy')
plt.plot(accuracy_results['C'], accuracy_results['Validation'], marker='o', label='Validation Accuracy')
plt.plot(accuracy_results['C'], accuracy_results['Test'], marker='o', label='Testing Accuracy')
plt.xlabel('C (Regularization Parameter)')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. C (RBF Kernel with Default Gamma)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print('\n\n--------------Multi-class Logistic Regression-------------------\n\n')
"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b.flatten(), jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')




# # Calculate and print category-wise errors for Logistic Regression
# def category_error(predicted_label, true_label, n_class):
#     total_error = {}
#     for i in range(n_class):
#         cls_indices = (true_label == i).ravel()
#         true_cls_lbls = true_label[cls_indices]
#         predicted_cls_lbls = predicted_label[cls_indices]
#         error = 1 - np.mean(predicted_cls_lbls == true_cls_lbls)
#         total_error[i] = error * 100
#     return total_error


# def printing_errors(errors, dataset_type):
#     for category, error in errors.items():
#         print(f'Error for Category {category}: {error:.4f}%')


# # Logistic Regression: Training Dataset & Test Dataset
# predicted_label = blrPredict(W, train_data)
# train_err = category_error(predicted_label, train_label, n_class)
# print('\n Training set Accuracy for Logistic Regression:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
# printing_errors(train_err, '')

# predicted_label = blrPredict(W, test_data)
# test_err = category_error(predicted_label, test_label, n_class)
# print('\n Testing set Accuracy Logistic Regression:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
# printing_errors(test_err, '')


# # Multi-class Logistic Regression: Training Dataset & Test Dataset
# predicted_label_b = mlrPredict(W_b, train_data)
# train_err_b = category_error(predicted_label_b, train_label, n_class)
# print('\n Training set Accuracy for Multi-class Logistic Regression: ' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')
# printing_errors(train_err_b, '')

# predicted_label_b = mlrPredict(W_b, test_data)
# test_err_b = category_error(predicted_label_b, test_label, n_class)
# print('\n Testing set Accuracy for Multi-class Logistic Regression: ' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
# printing_errors(test_err_b, '')


# models = ['Logistic Regression', 'Multi-class Logistic Regression']
# train_acc = [92.72, 93.44]
# val_acc = [91.47, 92.47]
# test_acc = [92.03, 92.55]

# x = range(len(models))

# plt.figure(figsize=(10, 6))
# plt.bar(x, train_acc, width=0.25, label='Training Accuracy', align='center', alpha=0.4, color='purple')
# plt.bar([p + 0.25 for p in x], val_acc, width=0.25, label='Validation Accuracy', align='center', alpha=0.4, color='cyan')
# plt.bar([p + 0.50 for p in x], test_acc, width=0.25, label='Testing Accuracy', align='center', alpha=0.4, color='orange')

# plt.xticks([p + 0.25 for p in x], models)
# plt.ylim(90, 94)  # Set Y-axis range from 90 to 95
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy Comparison: Logistic Regression vs. Multi-class Logistic Regression')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()