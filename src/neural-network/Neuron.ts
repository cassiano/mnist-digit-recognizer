import { HE_INIT_FACTOR } from '../constants'

/**
 * Represents a single neuron in the neural network.
 *
 * Each neuron has a set of weights (one per input connection) and a bias term.
 * The weights are initialized using He initialization, which is designed for
 * ReLU activations and helps prevent vanishing/exploding gradients.
 *
 * The forward computation for a neuron is: output = activation(sum(weight_i * input_i) + bias)
 */
export class Neuron {
  /** Weight vector — one weight per input connection */
  weights: number[]
  /** Bias term — added to the weighted sum before activation */
  bias: number

  constructor(inputCount: number) {
    this.bias = 0
    this.weights = Neuron.heInit(inputCount)
  }

  /**
   * He initialization: weights drawn from N(0, sqrt(2/n))
   *
   * This scales the initial weights by the square root of 2 divided by the
   * number of inputs, which maintains variance of activations across layers
   * when using ReLU. Uses the Box-Muller transform to generate normally
   * distributed random numbers from uniform random variables.
   */
  private static heInit(count: number): number[] {
    const scale = Math.sqrt(HE_INIT_FACTOR / count)
    const weights: number[] = new Array(count)

    for (let i = 0; i < count; i++) {
      // Box-Muller transform: generate standard normal from two uniform samples
      let u = 0

      while (u === 0) u = Math.random() // avoid log(0)

      const v = Math.random()

      weights[i] =
        Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale
    }

    return weights
  }
}
