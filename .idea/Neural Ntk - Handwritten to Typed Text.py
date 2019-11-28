from emnist import extract_training_samples
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

X, y = extract_training_samples('letters')
X = X / 255.

X_train, X_test = X[:20000], X[60000:70000]
y_train, y_test = y[:20000], y[60000:70000]

X_train = X_train.reshape(20000, 784)
X_test = X_test.reshape(10000, 784)

# img_index = 8080
# img = X_train[img_index]
# print(chr(y_train[img_index] + 96))
# plt.imshow(img.reshape((28, 28)))
# plt.show()


mlp1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=50, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
                     learning_rate_init=.1)

mlp1.fit(X_train, y_train)
print("Training set score: %f" % mlp1.score(X_train, y_train))
print("Test set score: %f" % mlp1.score(X_test, y_test))

mlp2 = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100,), max_iter=50, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
                     learning_rate_init=.1)
mlp2.fit(X_train, y_train)
print("Training set score: %f" % mlp2.score(X_train, y_train))
print("Test set score: %f" % mlp2.score(X_test, y_test))