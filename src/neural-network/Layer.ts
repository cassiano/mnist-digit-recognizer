import { Neuron } from './Neuron'
import { relu, reluDeriv, sigmoid, sigmoidDeriv, tanh, tanhDeriv, softmax } from './Activation'
import type { ActivationType } from './types'
import { timesForEach, timesMap } from '../utils'

/**
 * A single layer in the neural network.
 *
 * A layer consists of multiple neurons that each compute a weighted sum of
 * their inputs, add a bias, and apply an activation function. The layer
 * performs both forward propagation (computing outputs from inputs) and
 * backward propagation (computing gradients and updating weights).
 *
 * The layer stores intermediate values (inputs, preActivations, outputs)
 * needed for backpropagation.
 */
export class Layer {
  /** Array of neurons in this layer */
  neurons: Neuron[]
  /** Number of inputs to this layer (connections from previous layer) */
  inputSize: number
  /** Number of outputs from this layer (number of neurons) */
  outputSize: number
  /** Activation function applied to this layer's outputs */
  activation: ActivationType

  /** Inputs from the previous layer, stored for backpropagation */
  inputs: number[] = []
  /** Pre-activation values (z = Wx + b), stored for computing activation derivatives */
  preActivations: number[] = []
  /** Post-activation outputs, used as inputs to the next layer */
  outputs: number[] = []

  constructor(
    inputSize: number,
    outputSize: number,
    activation: ActivationType = 'relu',
  ) {
    this.inputSize = inputSize
    this.outputSize = outputSize
    this.activation = activation
    this.neurons = timesMap(outputSize, () => new Neuron(inputSize))
  }

  /**
   * Forward pass: computes layer outputs from inputs.
   *
   * For each neuron i:
   *   preActivation_i = bias_i + Σ(weight_ij * input_j)
   *   output_i = activation(preActivation_i)
   *
   * Stores inputs, preActivations, and outputs for use in backward().
   */
  forward(inputs: number[]): number[] {
    this.inputs = inputs
    this.preActivations = new Array(this.outputSize)

    // Compute weighted sum + bias for each neuron
    timesForEach(this.outputSize, i => {
      const neuron = this.neurons[i]
      let z = neuron.bias // z = Wx + b

      timesForEach(this.inputSize, j => {
        z += neuron.weights[j] * inputs[j]
      })

      this.preActivations[i] = z
    })

    // Apply activation function
    if (this.activation === 'softmax') {
      this.outputs = softmax(this.preActivations)
    } else if (this.activation === 'tanh') {
      this.outputs = this.preActivations.map(tanh)
    } else if (this.activation === 'sigmoid') {
      this.outputs = this.preActivations.map(sigmoid)
    } else {
      this.outputs = this.preActivations.map(relu)
    }

    return this.outputs
  }

  /**
   * Backward pass: computes gradients and updates weights/biases.
   *
   * For each neuron i:
   *   1. Compute local delta: delta_i = outputDelta_i * activation'(preActivation_i)
   *      - For softmax: delta = outputDelta (gradient flows directly through)
   *      - For ReLU: delta = outputDelta * 1 if preActivation > 0, else 0
   *      - For sigmoid: delta = outputDelta * sigmoid'(preActivation)
   *   2. Update weights: w_j -= learningRate * delta * input_j
   *   3. Update bias: b -= learningRate * delta
   *   4. Compute input deltas for previous layer: inputDelta_j += w_j * delta
   *
   * Returns input deltas to propagate gradients to the previous layer.
   */
  backward(outputDeltas: number[], learningRate: number): number[] {
    const inputDeltas = new Array(this.inputSize).fill(0)

    timesForEach(this.outputSize, i => {
      // Compute local gradient (delta) based on activation function
      let delta: number

      if (this.activation === 'softmax') {
        // Softmax + cross-entropy: gradient simplifies to (output - target)
        delta = outputDeltas[i]
      } else if (this.activation === 'tanh') {
        delta = outputDeltas[i] * tanhDeriv(this.preActivations[i])
      } else if (this.activation === 'sigmoid') {
        delta = outputDeltas[i] * sigmoidDeriv(this.preActivations[i])
      } else {
        // ReLU: derivative is 1 for positive pre-activation, 0 otherwise
        delta = outputDeltas[i] * reluDeriv(this.preActivations[i])
      }

      // Update weights and bias, accumulate input deltas for previous layer
      const neuron = this.neurons[i]

      timesForEach(this.inputSize, j => {
        inputDeltas[j] += neuron.weights[j] * delta

        // [/doc_img/Layer.ts/2026-09-09-18-44-06.png]
        neuron.weights[j] -= learningRate * delta * this.inputs[j]
      })

      // [/doc_img/Layer.ts/2026-09-09-18-56-56.png]
      neuron.bias -= learningRate * delta
    })

    return inputDeltas
  }
}
