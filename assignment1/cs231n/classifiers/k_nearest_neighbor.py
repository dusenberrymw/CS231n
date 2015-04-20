import numpy as np

class KNearestNeighbor:
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Input:
    X - A num_train x dimension array where each row is a training point.
    y - A vector of length num_train, where y[i] is the label for X[i, :]
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Input:
    X - A num_test x dimension array where each row is a test point.
    k - The number of nearest neighbors that vote for predicted label
    num_loops - Determines which method to use to compute distances
                between training points and test points.

    Output:
    y - A vector of length num_test, where y[i] is the predicted label for the
        test point X[i, :].
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Input:
    X - An num_test x dimension array where each row is a test point.

    Output:
    dists - A num_test x num_train array where dists[i, j] is the L2 distance
            between the ith test point and the jth training point.
            This has a geometric interpretation of computing the euclidean
            distance between two points. a^2 + b^ = c^2, where c is the
            distance between a and b.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]               #
        #####################################################################
        dists[i,j] = np.sqrt(np.sum(np.square(self.X_train[j,:] - X[i,:])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      # Use broadcasting to find rows of differences between all training
      # examples and the given ith test example, one row per training example,
      # and one column per dimension (pixel position for images).  Then square
      # the differences.  Then sum all of the columns to a single column, 
      # leaving a vector of summed differences from this test example, one per
      # training example.  Then take the sqrt of each value to complete the 
      # L2 distances from this test example. Store the L2 distances from the
      # ith test example in the ith row of dists.
      # Repeat for each test example in this loop.
      dists[i,:] = np.sqrt(np.sum(np.square(self.X_train - X[i,:]), axis=1))
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    # The L2 distance is the 2-norm of the difference between the points.
    # If we have point a and point b, then the L2 distance is:
    #   dist = sqrt(sum((a-b)^2)) 
    #
    # We can expand out the square function as follows:
    # dist  = sqrt(sum((a-b)(a-b)))
    #       = sqrt(sum(a^2 - 2ab + b^2))
    #
    # Then, we can apply the sum to each term:
    # dist  = sqrt(sum(a^2) - 2 * sum(ab) + sum(b^2)) 
    #
    # If a and b are instead matrices of examples, then the middle term will 
    # be sum(2ab'), which will be a matrix where every example in a is 
    # multiplied by every example in b using matrix multiplication to sum to 
    # single points.  So, this term will collapse each example a[i,:] * b[j,:]
    # to a single term in the resulting matrix, thus the summation is taken 
    # care of.
    #
    # Given this, for the a^2 and b^2 terms we can apply the square function
    # element-wise, and then take the resulting matrices, which will be the 
    # squared values of each example, and collapse each example down to a 
    # single value through summations.
    #
    # Then, we can broadcast these vectors to the 2ab' matrix.

    # For convenience, we assign X and self.X_train to a & b, respectively.
    a = X # i x k matrix = (i,k)
    b = self.X_train # j x k matrix = (j,k)

    # Need to add an axis to sum(a^2) so that dimensions line up to allow 
    # for broadcasting (sizes in each dimension must either be equal, or be 
    # set to 1, starting with the trailing dimension first).
    # The sum(b^2) term lines up in the trailing dimension (j), so it already
    # will broadcast
    sumasqr = np.sum(a**2, axis=1)[:,np.newaxis] # i x 1 matrix = (i,1)
    twoab = 2 * a.dot(b.T) # i x j matrix = (i,j)
    sumbsqr = np.sum(b**2, axis=1) # j length vector = (j)

    # Vectorized solution
    dists = np.sqrt(sumasqr - twoab + sumbsqr) # i x j matrix = (i, j)

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Input:
    dists - A num_test x num_train array where dists[i, j] gives the distance
            between the ith test point and the jth training point.

    Output:
    y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # training point, and use self.y_train to find the labels of these      #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      # Note: shortest distance is nearest neighbor
      sorted_neighbors = np.argsort(dists[i,:]) # return sorted indices
      k_nearest_neighbors = sorted_neighbors[0:k] # get top k indices
      closest_y = self.y_train[k_nearest_neighbors] # get top k labels
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      values, counts = np.unique(closest_y, return_counts=True) # label counts
      idx = np.argmax(counts) # get index of label with highest count
      y_pred[i] = values[idx] # set prediction as the label with highest count
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

