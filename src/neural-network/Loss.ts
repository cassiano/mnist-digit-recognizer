/**
 * Loss functions for measuring prediction error.
 *
 * Loss functions quantify how far the network's predictions are from the
 * true targets. Lower loss means better predictions. These are used to
 * compute gradients during backpropagation.
 */

/**
 * Cross-entropy loss for a single prediction.
 *
 * For classification, cross-entropy measures the negative log probability
 * of the true class: L = -log(p_target)
 *
 * The prediction is clipped to [epsilon, 1-epsilon] to avoid log(0) = -Infinity
 * which would occur if the network predicts probability 0 for the correct class.
 */
import { EPSILON } from '../constants'

export function crossEntropy(predicted: number[], actual: number): number {
  const clipped = Math.max(EPSILON, Math.min(1 - EPSILON, predicted[actual]))

  return -Math.log(clipped)
}

/**
 * Average cross-entropy loss over a batch of predictions.
 * Sums individual losses and divides by batch size.
 */
export function crossEntropyBatch(
  predictions: number[][],
  actuals: number[],
): number {
  const totalLoss = predictions.reduce(
    (acc, prediction, i) => acc + crossEntropy(prediction, actuals[i]),
    0,
  )

  return totalLoss / predictions.length
}

/**
 * Mean Squared Error (MSE) loss.
 * L = (1/n) * Σ(predicted_i - actual_i)²
 *
 * Commonly used for regression tasks. Not typically used for classification
 * because cross-entropy provides better gradient properties for softmax output.
 */
export function mse(predicted: number[], actual: number[]): number {
  const sum = predicted.reduce(
    (acc, value, i) => acc + (value - actual[i]) ** 2,
    0,
  )

  return sum / predicted.length
}
