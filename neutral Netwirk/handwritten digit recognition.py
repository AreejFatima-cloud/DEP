from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# FETCHING DATASETS
mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target'].astype(np.uint8)

# Convert DataFrame to NumPy array for easier indexing
x = x.to_numpy()

# Verifying shapes
    # (70000, 784) for MNIST
print('X shape:', x.shape)  

 # (70000,) for MNIST
print('Y shape:', y.shape) 

# Fetch an example digit
example_index = 36001
some_digit = x[example_index]

# Reshape it to plot
some_digit_image = some_digit.reshape(28, 28)

# Print the label for this digit
digit_label = y[example_index]
print(f"The label for the digit at index {example_index} is: {digit_label}")

# Plot the digit
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis('off')  # Turn off axis
plt.show()

# Splitting into training and test sets
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
# Creating a detector for digit '2'
y_train_2 = (y_train != 2)  # Binary labels for digit '2'
y_test_2 = (y_test != 2)    # Binary labels for digit '2'

# Verify the unique classes in the binary labelsxg
print("Unique classes in y_train_2:", np.unique(y_train_2))
print("Unique classes in y_test_2:", np.unique(y_test_2))

# Printing the boolean arrays with descriptive text
print("Boolean array for detecting '2' in the training set (first 100 elements):")
print(y_train_2[:100])  # Display first 100 elements for brevity

print("\nBoolean array for detecting '2' in the test set (first 100 elements):")
print(y_test_2[:100])  # Display first 100 elements for brevity

# Printing the counts with descriptive text
print("\nNumber of '2's in the training set: {}".format(np.sum(y_train_2)))
print("Number of '2's in the test set: {}".format(np.sum(y_test_2)))

# Ensure that the dataset contains more than one class
unique_classes_train = np.unique(y_train_2)
unique_classes_test = np.unique(y_test_2)
print(f"Unique classes in y_train_2: {unique_classes_train}")
print(f"Unique classes in y_test_2: {unique_classes_test}")

# Checking class distribution
print(f"Count of '2's in the training set: {np.sum(y_train_2)}")
print(f"Count of '2's in the test set: {np.sum(y_test_2)}")

# Creating the logistic regression model
clf = LogisticRegression(tol=0.1)

# Ensuring at least two classes are present
if len(unique_classes_train) > 1:
    clf.fit(x_train, y_train_2)
    some_digit_test = x_test[0]
    prediction = clf.predict([some_digit_test])
    print(f"Prediction for the first test digit: {prediction}")
else:
    print("Training data does not contain more than one class.")

# Using cross-validation to evaluate the model
if len(unique_classes_train) > 1:
    cv_scores = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores)}")
else:
    print("Cannot perform cross-validation because Training data does not contain more than one class.")
