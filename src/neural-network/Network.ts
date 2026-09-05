import { Layer } from './Layer'
import { timesMap } from '../utils'
import {
  DEFAULT_LEARNING_RATE,
  EPSILON,
  LEARNING_RATE_DECAY,
} from '../constants'
import type {
  NetworkConfig,
  TrainingData,
  TrainingResult,
  Prediction,
} from './types'

/**
 * Multi-layer perceptron (MLP) neural network.
 *
 * This is the core class that orchestrates forward/backward propagation
 * across multiple layers. It supports configurable architecture with
 * any number of hidden layers and neurons per layer.
 *
 * Architecture: Input(784) → Hidden₁ → Hidden₂ → ... → Output(10)
 * - Hidden layers use ReLU activation by default
 * - Output layer always uses softmax for probability distribution
 *
 * Training uses stochastic gradient descent (SGD) with mini-batch updates
 * and learning rate decay.
 */
export class Network {
  /** Ordered list of layers from input to output */
  layers: Layer[]
  /** Current learning rate (decays during training) */
  learningRate: number

  constructor(config: NetworkConfig) {
    this.learningRate = config.learningRate
    this.layers = []

    // Create layers: for each consecutive pair of sizes, create a layer
    const layerSizes = config.layers

    for (let i = 0; i < layerSizes.length - 1; i++) {
      const isOutput = i === layerSizes.length - 2

      this.layers.push(
        new Layer(
          layerSizes[i],
          layerSizes[i + 1],
          isOutput ? 'softmax' : config.activation || 'relu',
        ),
      )
    }
  }

  /**
   * Forward pass through the entire network.
   * Chains layer.forward() calls, feeding each layer's output as input to the next.
   * Returns the final output (probability distribution over 10 digits).
   */
  forward(input: number[]): number[] {
    let current = input

    for (const layer of this.layers) current = layer.forward(current)

    return current
  }

  /**
   * Makes a prediction on a single input.
   * Runs forward pass, then finds the digit with highest probability.
   * Returns the predicted digit, its confidence, and full probability distribution.
   */
  predict(input: number[]): Prediction {
    const probabilities = this.forward(input)
    let maxIdx = 0
    let maxVal = probabilities[0]

    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxVal) {
        maxVal = probabilities[i]
        maxIdx = i
      }
    }

    return { digit: maxIdx, confidence: maxVal, probabilities }
  }

  /**
   * Backward pass through the entire network.
   *
   * 1. Computes output layer error: delta = output - one_hot(target)
   *    This is the gradient of cross-entropy loss w.r.t. softmax pre-activations.
   * 2. Propagates error backwards through each layer using layer.backward()
   *
   * @param target - The true label (0-9)
   * @param learningRate - Optional override for this update's learning rate
   */
  backward(target: number, learningRate?: number): void {
    const lr = learningRate ?? this.learningRate
    const outputLayer = this.layers[this.layers.length - 1]

    // Output layer error: difference between prediction and one-hot target
    const outputDeltas = new Array(outputLayer.outputSize)

    for (let i = 0; i < outputLayer.outputSize; i++)
      outputDeltas[i] = outputLayer.outputs[i] - (i === target ? 1 : 0)

    // Propagate deltas backwards through all layers
    let deltas = outputDeltas

    for (let l = this.layers.length - 1; l >= 0; l--)
      deltas = this.layers[l].backward(deltas, lr)
  }

  /**
   * Trains the network on the provided dataset.
   *
   * For each epoch:
   *   1. Shuffles training data indices
   *   2. For each sample: forward pass, compute accuracy, backward pass
   *   3. Decays learning rate every batchSize samples
   *   4. Calls onEpoch callback with training results
   *
   * Note: This method is synchronous and runs training in the main thread.
   * The TrainingPanel component implements its own async training loop with
   * chunked processing to keep the UI responsive.
   */
  train(
    data: TrainingData,
    epochs: number,
    batchSize: number = 32,
    onEpoch?: (result: TrainingResult) => void,
  ): void {
    const { inputs, labels } = data

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0
      let correct = 0

      // Fisher-Yates shuffle for random sampling
      const indices = timesMap(inputs.length, i => i)
      this.shuffle(indices)

      for (let i = 0; i < indices.length; i++) {
        const idx = indices[i]
        const output = this.forward(inputs[idx])
        const predicted = output.indexOf(Math.max(...output))

        if (predicted === labels[idx]) correct++

        // Cross-entropy loss: -log(p_target)
        const targetVal = output[labels[idx]]
        totalLoss += -Math.log(Math.max(targetVal, EPSILON))

        this.backward(labels[idx])

        // Learning rate decay: reduce by 0.1% every batchSize samples
        if (i % batchSize === 0 && i > 0) this.learningRate *= LEARNING_RATE_DECAY
      }

      const accuracy = correct / inputs.length
      const avgLoss = totalLoss / inputs.length

      onEpoch?.({ epoch: epoch + 1, loss: avgLoss, accuracy })
    }
  }

  /**
   * Fisher-Yates (Knuth) shuffle: randomizes array in-place in O(n).
   * Used to ensure each epoch sees training data in a different order.
   */
  private shuffle(array: number[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))

      ;[array[i], array[j]] = [array[j], array[i]]
    }
  }

  /**
   * Returns the outputs of a specific layer (after activation).
   * Used by NetworkVisualization to show neuron activations.
   * Returns a copy to prevent external mutation.
   */
  getLayerOutputs(layerIndex: number): number[] {
    if (layerIndex >= 0 && layerIndex < this.layers.length)
      return [...this.layers[layerIndex].outputs]

    return []
  }

  /**
   * Computes summary statistics for the network architecture.
   * Returns total neuron count, total parameter count, and per-layer details.
   */
  getNetworkInfo(): {
    totalNeurons: number
    totalWeights: number
    layerDetails: { neurons: number; weights: number }[]
  } {
    let totalNeurons = 0
    let totalWeights = 0
    const layerDetails: { neurons: number; weights: number }[] = []

    for (const layer of this.layers) {
      // Parameters = weights (inputSize * outputSize) + biases (outputSize)
      const layerWeights = layer.outputSize * layer.inputSize + layer.outputSize

      totalNeurons += layer.outputSize
      totalWeights += layerWeights

      layerDetails.push({ neurons: layer.outputSize, weights: layerWeights })
    }

    return { totalNeurons, totalWeights, layerDetails }
  }

  /**
   * Serializes the network to a JSON string for localStorage persistence.
   * Saves layer sizes, activations, and all neuron weights/biases.
   */
  serialize(): string {
    const data = {
      layerSizes: this.layers.map(l => [
        l.inputSize,
        l.outputSize,
        l.activation,
      ]),
      weights: this.layers.map(l =>
        l.neurons.map(n => ({ w: [...n.weights], b: n.bias })),
      ),
    }

    return JSON.stringify(data)
  }

  /**
   * Restores a network from a serialized JSON string.
   * Rebuilds the architecture and loads all weights/biases.
   */
  static deserialize(json: string): Network {
    const data = JSON.parse(json)
    const sizes = [
      data.layerSizes[0][0],
      ...data.layerSizes.map((l: number[]) => l[1]),
    ]
    const net = new Network({
      layers: sizes,
      learningRate: DEFAULT_LEARNING_RATE,
      activation: 'relu',
    })

    for (let l = 0; l < data.weights.length; l++) {
      for (let n = 0; n < data.weights[l].length; n++) {
        net.layers[l].neurons[n].weights = data.weights[l][n].w
        net.layers[l].neurons[n].bias = data.weights[l][n].b
      }
    }

    return net
  }
}
