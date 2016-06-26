import numpy as np
from random import shuffle
import scipy.sparse
from IPython import embed

class SoftmaxClassifier:

  def __init__(self):
    self.theta = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train the classifier using mini-batch stochastic gradient descent.

    Inputs:
    - X: m x d array of training data. Each training point is a d-dimensional
         row.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train,dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    self.theta = np.random.randn(dim,num_classes) * 0.001
    print("learning_rate = ", learning_rate, "reg = ", reg)
    if (learning_rate == .00005 and reg == 100000000):
        print("nan and inf is coming")
        # embed()
    # Run stochastic gradient descent to optimize theta
    loss_history = []
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      # Hint: 3 lines of code expected
      # indices = np.random.choice(num_train, batch_size)
      # X_batch = X[indices,:]
      X_batch = X
      # y_batch = y[indices]
      y_batch = y
      #embed()
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      # Hint: 1 line of code expected
      self.theta -= learning_rate * grad

      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: m x d array of training data. Each row is a d-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length m, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    # Hint: 1 line of code expected
    y_pred = np.argmax(np.dot(X,self.theta),axis = 1)

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: m x d array of data; each row is a data point.
    - y_batch: 1-dimensional array of length m with labels 0...K-1, for K classes.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.theta; an array of the same shape as theta
    """
    #return softmax_loss_naive(self.theta, X_batch, y_batch, reg)
    return softmax_loss_vectorized(self.theta, X_batch, y_batch, reg)

  
def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  # Hint: about 5-10 lines of code expected
  de = np.dot(theta.T, X.T).T
  de = de.T - np.max(de,axis=1)
  den = np.exp(de)

  for i in range(0, m):
      for k in range(0, theta.shape[1]):
            # embed()
          J += (1 if k == y[i] else 0) * np.log(float(den[k,i]) / np.sum(den[:,i]) )

  # embed()
  J = -(J / m) + (reg/(2*m)) * np.sum(np.square(theta))

  for k in range(0,theta.shape[1]):
      for i in range(0, m):
          P = den[k,i] / np.sum(den[:,i])
          grad[:,k] += -X[i] * ((1 if k == y[i] else 0) - P)/m + (reg/m) * theta[:,k]
          
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

def convert_y_to_matrix(y):
  """
  convert an array of m elements with values in {0,...,K-1} to a boolean matrix
  of size m x K where there is a 1 for the value of y in that row.

  """
  y = np.array(y)
  data = np.ones(len(y))
  indptr = np.arange(len(y)+1)
  mat = scipy.sparse.csr_matrix((data,y,indptr))
  return mat.todense()

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  # Hint: 4-6 lines of code expected
  
  K = theta.shape[1]
  In = np.zeros((m,K))
  rows = np.ndarray.astype(np.linspace(0,m-1,m),int)
  In[rows,y] = 1
  de = np.dot(X,theta)
  den = np.exp((de - np.max(de,axis=1)[:,np.newaxis]))
  J = - (1./m) * np.sum( np.log(np.sum(np.multiply(In,den),axis=1) / np.sum(den,axis=1))) + (reg/(2*m)) * np.sum(np.square(theta))
  grad = -(1./m) * np.dot(X.T, In + np.divide(-den, np.sum(den,axis=1)[:,np.newaxis])) + (reg/m) * theta
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
