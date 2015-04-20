# As usual, a bit of setup
import cPickle as pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.classifiers.convnet import *
from cs231n.data_utils import load_CIFAR10
from cs231n.vis_utils import visualize_grid

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    x_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_val, y_val, X_test, y_test


# Get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

print X_train.dtype
print X_val.dtype


# # Experiment!
# Experiment and try to get the best performance that you can on CIFAR-10 using a ConvNet. Here are some ideas to get you started:
# 
# ### Things you should try:
# - Filter size: Above we used 7x7; this makes pretty pictures but smaller filters may be more efficient
# - Number of filters: Above we used 32 filters. Do more or fewer do better?
# - Network depth: The network above has two layers of trainable parameters. Can you do better with a deeper network? You can implement alternative architectures in the file `cs231n/classifiers/convnet.py`. Some good architectures to try include:
#     - [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
#     - [conv-relu-pool]XN - [affine]XM - [softmax or SVM]
#     - [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
# 
# ### Tips for training
# For each network architecture that you try, you should tune the learning rate and regularization strength. When doing this there are a couple important things to keep in mind:
# 
# - If the parameters are working well, you should see improvement within a few hundred iterations
# - Remember the course-to-fine approach for hyperparameter tuning: start by testing a large range of hyperparameters for just a few training iterations to find the combinations of parameters that are working at all.
# - Once you have found some sets of parameters that seem to work, search more finely around these parameters. You may need to train for more epochs.
# 
# ### Going above and beyond
# If you are feeling adventurous there are many other features you can implement to try and improve your performance. You are **not required** to implement any of these; however they would be good things to try for extra credit.
# 
# - Alternative update steps: For the assignment we implemented SGD+momentum and RMSprop; you could try alternatives like AdaGrad or AdaDelta.
# - Other forms of regularization such as L1 or Dropout
# - Alternative activation functions such as leaky ReLU or maxout
# - Model ensembles
# - Data augmentation
# 
# ### What we expect
# At the very least, you should be able to train a ConvNet that gets at least 65% accuracy on the validation set. This is just a lower bound - if you are careful it should be possible to get accuracies much higher than that! Extra credit points will be awarded for particularly high-scoring models or unique approaches.
# 
# You should use the space below to experiment and train your network. The final cell in this notebook should contain the training, validation, and test set accuracies for your final trained network. In this notebook you should also write an explanation of what you did, any additional features that you implemented, and any visualizations or graphs that you make in the process of training and evaluating your network.
# 
# Have fun and happy training!

# TODO: Train a ConvNet to do really well on CIFAR-10!

# Keep best model across entire hyperparameter search
# Try to load previous best
file_name = 'best.pickle'
try:
  best = pickle.load(open(file_name, 'rb'))
  print best['model']['W1'].dtype
except IOError:
  # No previous best
  print "Using new best"
  best = {}
  best['model'] = None
  best['val_acc'] = -1
except Exception, e:
  print 'failed to load best model & settings from %s with error:' % (file_name)
  print e

tries = 1
results = {} # contains each combination of hyperparameters for this set of tries
loss_function = two_layer_convnet

tic = time.time()
for _ in xrange(tries):
    # randomly generate hyperparameters -- start with a coarse range, then fine-tune    
    hyperparameters = {}
    init_hyp = {}
    train_hyp = {}
    init_hyp['filter_size'] = 3
    init_hyp['num_filters'] = random.randint(80, 100)
    train_hyp['learning_rate'] = 10 ** random.uniform(-4, -3)
    train_hyp['reg'] = 10 ** random.uniform(-8, -7)
    train_hyp['num_epochs'] = 0.5
    train_hyp['momentum'] = 0.9
    train_hyp['batch_size'] = 100
    train_hyp['acc_frequency'] = 100
    hyperparameters['init'] = init_hyp
    hyperparameters['train'] = train_hyp

    # train model
    model = init_two_layer_convnet(dtype=X_train.dtype, **hyperparameters['init'])
    trainer = ClassifierTrainer()
    model, loss_history, train_acc_history, val_acc_history = trainer.train(
                  X_train, y_train, X_val, y_val, model, loss_function,
                  **hyperparameters['train'])

    # store results
    train_acc = max(train_acc_history) # the model returned corresponds to the best accuracy
    val_acc = max(val_acc_history)
    results[(val_acc, train_acc)] = hyperparameters
    if val_acc > best['val_acc']:
        best['model'] = model
        best['hyperparameters'] = hyperparameters
        best['val_acc'] = val_acc
        best['train_acc'] = train_acc
        best['val_acc_history'] = val_acc_history
        best['train_acc_history'] = train_acc_history
        best['loss_history'] = loss_history

toc = time.time()

# dump best to file
file_name = 'best.pickle'
try:
  pickle.dump(best, open(file_name, 'w'))
except Exception, e:
  print 'failed to dump best model & settings into %s with error:' % (file_name)
  print e

# Print results
for val_acc, train_acc in sorted(results):
    hyperparameters = results[(val_acc, train_acc)]
    print 'val accuracy: {:.3f} train accuracy: {:.3f}, fs {init[filter_size]}, nf {init[num_filters]:>3}, '          'lr {train[learning_rate]:.4e}, reg {train[reg]:.4e}, epochs {train[num_epochs]:.2f}'.format(
                val_acc, train_acc, **hyperparameters)

print
print 'Best overall validation accuracy achieved during cross-validation: %f' % best['val_acc']
print 'Best hyperparameters: fs {init[filter_size]}, nf {init[num_filters]}, '\
        'lr {train[learning_rate]:.4e}, reg {train[reg]:.4e}, epochs {train[num_epochs]:.2f}'.format(
                val_acc, train_acc, **hyperparameters)
print 'Training took %fm (%fs)' % ((toc-tic)/60, toc-tic)

# Plot the loss function and train / validation accuracies for best model
plt.subplot(2, 1, 1)
plt.plot(best['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(best['train_acc_history'])
plt.plot(best['val_acc_history'])
plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
plt.xlabel('Check')
plt.ylabel('Clasification accuracy')
plt.show()


# # Visualize weights
# We can visualize the convolutional weights from the first layer. If 
# everything worked properly, these will usually be edges and blobs of 
# various colors and orientations.
grid = visualize_grid(best['model']['W1'].transpose(0, 2, 3, 1))
plt.imshow(grid.astype('uint8'))
plt.show()

