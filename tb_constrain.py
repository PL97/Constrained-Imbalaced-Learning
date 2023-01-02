import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import math
import numpy as np
from six.moves import xrange


def create_data(dimension = 10):
    # Create a simulated 10-dimensional training dataset consisting of 1000 labeled
    # examples, of which 800 are labeled correctly and 200 are mislabeled.
    num_examples = 1000
    num_mislabeled_examples = 100

    # Create random "ground truth" parameters for a linear model.
    ground_truth_weights = np.random.normal(size=dimension) / math.sqrt(dimension)
    ground_truth_threshold = 0

    # Generate a random set of features for each example.
    features = np.random.normal(size=(num_examples, dimension)).astype(
        np.float32) / math.sqrt(dimension)
    # Compute the labels from these features given the ground truth linear model.
    labels = (np.matmul(features, ground_truth_weights) >
            ground_truth_threshold).astype(np.float32)
    # Add noise by randomly flipping num_mislabeled_examples labels.
    mislabeled_indices = np.random.choice(
        num_examples, num_mislabeled_examples, replace=False)
    labels[mislabeled_indices] = 1 - labels[mislabeled_indices]

    # Constant Tensors containing the labels and features.
    constant_labels = tf.constant(labels, dtype=tf.float32)
    constant_features = tf.constant(features, dtype=tf.float32)
    
    
    
    return constant_features, constant_labels, features, labels


def average_hinge_loss(labels, predictions):
    # Recall that the labels are binary (0 or 1).
    signed_labels = (labels * 2) - 1
    return np.mean(np.maximum(0.0, 1.0 - signed_labels * predictions))

def recall(labels, predictions):
    # Recall that the labels are binary (0 or 1).
    positive_count = np.sum(labels)
    true_positives = labels * (predictions > 0)
    true_positive_count = np.sum(true_positives)
    return true_positive_count / positive_count

def precision(labels, predictions):
    # Recall that the labels are binary (0 or 1).
    predict_positive_count = np.sum(predictions>0)
    true_positives = labels * (predictions > 0)
    true_positive_count = np.sum(true_positives)
    return true_positive_count / predict_positive_count


def predictions():
    return tf.tensordot(constant_features, weights, axes=(1, 0)) - threshold

if __name__ == "__main__":
    dimension = 10
    constant_features, constant_labels, features, labels = create_data(dimension = dimension)
    # Create variables containing the model parameters.
    weights = tf.Variable(tf.zeros(dimension), dtype=tf.float32, name="weights")
    threshold = tf.Variable(0.0, dtype=tf.float32, name="threshold")
    
    context = tfco.rate_context(predictions, labels=lambda: constant_labels)
    
    # We will constrain the recall to be at least 90%.
    print("======== constrained performance ==============")
    recall_lower_bound = 0.7
    # mp = tfco.RateMinimizationProblem(
    #     tfco.error_rate(context), [tfco.recall(context) >= recall_lower_bound])
    # numerator, denominator = tfco.precision_ratio(context)
    objective = tfco.recall(context)
    
    mp = tfco.RateMinimizationProblem(
        objective, [tfco.precision(context) >= 0.9])
    opt = tfco.ProxyLagrangianOptimizerV1(tf.compat.v1.train.AdagradOptimizer(1.0))
    opt.minimize(mp)
    

    for ii in xrange(10000):
        opt.minimize(mp)

    trained_weights = weights.numpy()
    trained_threshold = threshold.numpy()
    
    
    trained_predictions = np.matmul(features, trained_weights) - trained_threshold
    print("Constrained average hinge loss = %f" %
        average_hinge_loss(labels, trained_predictions))
    print("Constrained precision = %f " % precision(labels, trained_predictions))
    print("Constrained recall = %f" % recall(labels, trained_predictions))
    
    
    
    ## unconstrained performance
    print("======== unconstrained performance ==============")
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=1.0)
    var_list = [weights, threshold]
    
    for ii in xrange(1000):
        # For optimizing the unconstrained problem, we just minimize the "objective"
        # portion of the minimization problem.
        optimizer.minimize(mp.objective, var_list=var_list)
    
    trained_weights = weights.numpy()
    trained_threshold = threshold.numpy()

    trained_predictions = np.matmul(features, trained_weights) - trained_threshold
    print("Unconstrained average hinge loss = %f" % average_hinge_loss(
        labels, trained_predictions))
    print("Constrained precision = %f " % precision(labels, trained_predictions))
    print("Unconstrained recall = %f" % recall(labels, trained_predictions))