// Standalone test — uses the Network class from the project.
// Test: does forward + backward produce non-NaN, and does training reduce loss?

// To run it: `npx tsx test-nn.ts`

import { Network } from './src/neural-network/Network'

// ── Tiny network: 3 → 4 → 2 ────────────────────────────────────────────────
const tiny = new Network({
  layers: [3, 4, 2],
  learningRate: 0.1,
  activation: 'relu',
})

const trainingData = [
  { input: [1, 0, 0], label: 0 },
  { input: [0, 1, 0], label: 1 },
  { input: [1, 0, 0], label: 0 },
  { input: [0, 1, 0], label: 1 },
]

function hasNaN() {
  for (const layer of tiny.layers) {
    for (const n of layer.neurons) {
      for (const w of n.weights) {
        if (!Number.isFinite(w)) return true
      }
    }
  }
  return false
}

// Check initial forward
console.log('=== Initial forward ===')
for (const d of trainingData) {
  const result = tiny.predict(d.input)
  console.log(
    `Input: [${d.input}] -> probs: [${result.probabilities.map(v => v.toFixed(4))}] NaN: ${result.probabilities.some(v => !Number.isFinite(v))}`,
  )
}

// Train 100 epochs
for (let epoch = 0; epoch < 100; epoch++) {
  let totalLoss = 0
  let correct = 0

  for (const d of trainingData) {
    const result = tiny.predict(d.input)
    const loss = -Math.log(Math.max(result.probabilities[d.label], 1e-15))

    totalLoss += loss
    if (result.digit === d.label) correct++

    tiny.backward(d.label)
  }

  if (epoch % 20 === 0) {
    console.log(
      `Epoch ${epoch}: loss=${(totalLoss / trainingData.length).toFixed(4)} acc=${correct}/${trainingData.length} hasNaN=${hasNaN()}`,
    )
  }
}

// Check final forward
console.log('\n=== Final forward ===')
for (const d of trainingData) {
  const result = tiny.predict(d.input)
  console.log(
    `Input: [${d.input}] -> probs: [${result.probabilities.map(v => v.toFixed(4))}] predicted=${result.digit} target=${d.label} NaN: ${result.probabilities.some(v => !Number.isFinite(v))}`,
  )
}

// ── MNIST-sized: 784 → 16 → 16 → 10 ────────────────────────────────────────
console.log('\n=== MNIST-sized test ===')
const mnist = new Network({
  layers: [784, 16, 16, 10],
  learningRate: 0.1,
  activation: 'relu',
})

const mnistData = Array.from({ length: 10 }, (_, s) => ({
  input: Array.from({ length: 784 }, () => Math.random()),
  label: s,
}))

function mnistHasNaN() {
  for (const layer of mnist.layers) {
    for (const n of layer.neurons) {
      for (const w of n.weights) {
        if (!Number.isFinite(w)) return true
      }
    }
  }
  return false
}

// Forward check
console.log('Initial forward:')
const initResult = mnist.predict(mnistData[0].input)
console.log(
  `  NaN in output: ${initResult.probabilities.some(v => !Number.isFinite(v))}`,
)
console.log(`  NaN in weights: ${mnistHasNaN()}`)
console.log(`  Output: [${initResult.probabilities.map(v => v.toFixed(4))}]`)

// Train 50 epochs on all 10 samples
for (let epoch = 0; epoch < 50; epoch++) {
  let correct = 0

  for (const d of mnistData) {
    const result = mnist.predict(d.input)

    if (result.digit === d.label) correct++

    mnist.backward(d.label)
  }

  if (epoch % 10 === 0) {
    console.log(`Epoch ${epoch}: acc=${correct}/10 hasNaN=${mnistHasNaN()}`)
  }
}

// Final check
console.log('\nFinal forward:')
for (const d of mnistData) {
  const result = mnist.predict(d.input)
  console.log(
    `  target=${d.label} predicted=${result.digit} probs=[${result.probabilities.map(v => v.toFixed(3))}] NaN=${result.probabilities.some(v => !Number.isFinite(v))}`,
  )
}

console.log(`\nFinal NaN in weights: ${mnistHasNaN()}`)
