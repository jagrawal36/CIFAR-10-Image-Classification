import random
import numpy as np
import utils
from softmax import softmax_loss_naive, softmax_loss_vectorized
from softmax import SoftmaxClassifier
import time
import itertools
import scipy.ndimage as si
from IPython import embed
import pandas as pd
df = pd.read_csv('../trainLabels.csv')

class_labels = {'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}
class_inverted = dict((v, k) for k, v in class_labels.iteritems())
df["value"] = [class_labels[row.label] for idx,row in df.iterrows()]
y_raw = np.asarray(df["value"])
y_train = y_raw[0:10000]
y_val = y_raw[10000:11000]

X_train = []
for i in range(1,10001):
  file = "../train/"+str(i)+".png"
  x = si.imread(file)
  x = x.reshape(3072)
  X_train.append(x)
  if i % 5000 == 0:
    print("Reading data for index = ", i)
X_train = np.asarray(X_train)

X_val = []
for i in range(10001,11001):
  file = "../train/"+str(i)+".png"
  x = si.imread(file)
  x = x.reshape(3072)
  X_val.append(x)
  if i % 5000 == 0:
    print("Reading data for index = ", i)
X_val = np.asarray(X_val)


X_train,_,_ = utils.std_features(X_train)
X_train = np.vstack((np.ones(X_train.shape[0]), X_train.T)).T

X_val,_,_ = utils.std_features(X_val)
X_val = np.vstack((np.ones(X_val.shape[0]), X_val.T)).T

theta = np.random.randn(3073,10) * 0.0001
loss, grad = softmax_loss_naive(theta, X_train, y_train, 0.0)

# Loss should be something close to - log(0.1)

print 'loss:', loss, ' should be close to ', - np.log(0.1)

tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(theta, X_train, y_train, 0.00001)
toc = time.time()
print 'naive loss: %e computed in %fs' % (loss_naive, toc - tic)

tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(theta, X_train, y_train, 0.00001)
toc = time.time()
print 'vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)

# We use the Frobenius norm to compare the two versions
# of the gradient.

grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print 'Loss difference: %f' % np.abs(loss_naive - loss_vectorized)
print 'Gradient difference: %f' % grad_difference

results = {}
best_val = -1
best_softmax = None
#learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
#regularization_strengths = [ 5e4, 1e5, 5e5, 1e8]
learning_rates = [ 5e-6]
regularization_strengths = [1e5]

cl = SoftmaxClassifier()
loss_hist = []
for lr,rs in itertools.product(learning_rates,regularization_strengths):
    _ = cl.train(X_train, y_train, lr, rs, num_iters = 4000, batch_size = 400, verbose = True)
    loss,_ = cl.loss(X_val,y_val, rs)
    pred_t = cl.predict(X_train)
    pred_v = cl.predict(X_val)
    #embed()
    train_match = np.where(pred_t == y_train)
    train_accuracy = float(len(train_match[0]))/len(y_train)
    val_match = np.where(pred_v == y_val)
    val_accuracy = float(len(val_match[0]))/len(y_val)
    results[(lr,rs)] = (train_accuracy,val_accuracy)
    loss_hist.append(loss)
    # print("For lr,rs = ",lr,rs, "Loss  value = ",loss)

#embed()
ind = np.where(loss_hist == np.min(loss_hist))[0][0]
parameters = list(itertools.product(learning_rates,regularization_strengths))[ind]
best_softmax = SoftmaxClassifier()
_ = best_softmax.train(X_train, y_train, parameters[0], parameters[1], num_iters = 4000, batch_size = 400, verbose = True)
best_val = np.max(np.asarray(results.values())[:,1])
  
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val

# Evaluate the best softmax classifier on test set
X_test = []
for i in range(1,300001):
  file = "../test/"+str(i)+".png"
  x = si.imread(file)
  x = x.reshape(3072)
  X_test.append(x)
  if i %5000 == 0:
    print("Reading data for index = ", i)

X_test = np.asarray(X_test)

mu = np.mean(X_test,axis = 0)
sigma = np.std(X_test,axis = 0)

y_pred = []
for i in range(0,30):
  xx = ( X_test[i*10000:(i+1)*10000] - mu )/ sigma
  xx = np.vstack((np.ones(xx.shape[0]), xx.T)).T
  y_pred.append(best_softmax.predict(xx))

y_test_pred = np.asarray(y_pred).flatten()
y_class = [class_inverted[row] for row in y_test_pred]
out = pd.DataFrame(y_class)
out.index += 1
out.to_csv('out_softmax.csv')

