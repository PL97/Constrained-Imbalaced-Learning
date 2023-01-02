import numpy as np
import math
import torch
import matplotlib.pyplot as plt

def generate_data(dimension = 10, device=None):
    # Create a simulated 10-dimensional training dataset consisting of 1000 labeled
    # examples, of which 800 are labeled correctly and 200 are mislabeled.
    num_examples = 1000
    num_mislabeled_examples = 200

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
    labels = labels.reshape(-1, 1)

    # Constant Tensors containing the labels and features.
    tensor_features = torch.from_numpy(features).to(device)
    tensor_labels = torch.from_numpy(labels).to(device)
    
    ## plot the data
    if dimension == 2:
        positive_idx = np.where(labels==1)
        negative_idx = np.where(labels!=1)
        plt.scatter(features[positive_idx, 0], features[positive_idx, 1], color='green', label='positive')
        plt.scatter(features[negative_idx, 0], features[negative_idx, 1], color='red', label='negative')
        plt.savefig("data_orig.png")
    
    return tensor_features, tensor_labels, features, labels
