/**
 * Type definitions for the neural network module.
 *
 * These types define the shape of configuration objects, training data,
 * and prediction results used throughout the neural network.
 */

/** Supported activation function types */
export type ActivationType = 'relu' | 'sigmoid' | 'softmax' | 'none';

/**
 * Configuration for creating a new neural network.
 * @property layers - Array of layer sizes, e.g. [784, 16, 16, 10] for input→hidden→output
 * @property learningRate - Step size for gradient descent updates
 * @property activation - Activation function for hidden layers (default: 'relu')
 */
export interface NetworkConfig {
  layers: number[];
  learningRate: number;
  activation?: ActivationType;
}

/**
 * Training dataset containing input features and corresponding labels.
 * @property inputs - 2D array where each row is a flattened image (784 values)
 * @property labels - 1D array of digit labels (0-9)
 */
export interface TrainingData {
  inputs: number[][];
  labels: number[];
}

/**
 * Result after completing one training epoch.
 * @property epoch - Current epoch number (1-indexed)
 * @property loss - Average loss over the epoch
 * @property accuracy - Fraction of correct predictions on test data
 */
export interface TrainingResult {
  epoch: number;
  loss: number;
  accuracy: number;
}

/**
 * Prediction result from the network.
 * @property digit - Predicted digit (0-9)
 * @property confidence - Probability assigned to the predicted digit
 * @property probabilities - Full probability distribution over all 10 digits
 */
export interface Prediction {
  digit: number;
  confidence: number;
  probabilities: number[];
}
