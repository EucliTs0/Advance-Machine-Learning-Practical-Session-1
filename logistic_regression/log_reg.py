from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
import matplotlib.pyplot as plt
import time

# dataset
n_samples = 2000
#dataset = datasets.make_classification(n_samples, n_features, random_state=1)   # classification
dataset = datasets.make_circles(n_samples)      #circles
#dataset = datasets.make_moons(n_samples)         #moons

X_train = dataset[0][0:1000]
Y_train = dataset[1][0:1000]
X_test = dataset[0][1000:2000]
Y_test = dataset[1][1000:2000]


# initialize models
lr = LogisticRegression()
klr = LogisticRegression()

# fit normal
# start_time = time.time()
# lr.fit(X_train,Y_train)
# print round(time.time() - start_time, 5)
# start_time = time.time()
# pred = lr.predict(X_test)
# print round(time.time() - start_time, 5)
# print lr.score(X_test, Y_test)
# compute gram and fit
deg = 3
gamma = 15

start_time = time.time()
#gram_train = polynomial_kernel(X_train, degree=deg)
gram_train = rbf_kernel(X_train, gamma=gamma)
klr.fit(gram_train, Y_train)
print round(time.time() - start_time, 5)

start_time = time.time()
#gram_test = polynomial_kernel(X_test,X_train, degree=deg)
gram_test = rbf_kernel(X_test,X_train, gamma=gamma)
pred = klr.predict(gram_test)
print round(time.time() - start_time, 5)
print klr.score(gram_test, Y_test)

# plt.figure()
# plt.plot(X_test[pred == 1,0], X_test[pred == 1,1],'ro')
# plt.plot(X_test[pred == 0,0], X_test[pred == 0,1],'bo')
# plt.xlabel('x_values')
# plt.ylabel('y_values')
# plt.title('Logistic Regression - RBF - gamma = 15')
# plt.show()
