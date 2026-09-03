// Standalone test - no React/browser dependencies
// Test: does forward + backward produce non-NaN, and does training reduce loss?

function heInit(count) {
  const scale = Math.sqrt(2 / count);
  const weights = new Array(count);
  for (let i = 0; i < count; i++) {
    let u = 0;
    while (u === 0) u = Math.random();
    const v = Math.random();
    weights[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
  }
  return weights;
}

function relu(x) { return x > 0 ? x : 0; }
function reluDeriv(x) { return x > 0 ? 1 : 0; }
function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// Create a tiny network: 3 -> 4 -> 2
const layers = [
  { neurons: [], inputSize: 3, outputSize: 4 },
  { neurons: [], inputSize: 4, outputSize: 2, softmax: true },
];

for (const layer of layers) {
  for (let i = 0; i < layer.outputSize; i++) {
    layer.neurons.push({ weights: heInit(layer.inputSize), bias: 0 });
  }
}

function forward(input) {
  let current = input;
  const cache = [];
  for (const layer of layers) {
    const preActivations = [];
    for (let i = 0; i < layer.outputSize; i++) {
      const n = layer.neurons[i];
      let sum = n.bias;
      for (let j = 0; j < layer.inputSize; j++) {
        sum += n.weights[j] * current[j];
      }
      preActivations.push(sum);
    }
    let outputs;
    if (layer.softmax) {
      outputs = softmax(preActivations);
    } else {
      outputs = preActivations.map(relu);
    }
    cache.push({ inputs: current, preActivations, outputs });
    current = outputs;
  }
  return { output: current, cache };
}

function backward(target, lr, cache) {
  const outputLayer = layers[layers.length - 1];
  const outputDeltas = new Array(outputLayer.outputSize);
  for (let i = 0; i < outputLayer.outputSize; i++) {
    outputDeltas[i] = cache[layers.length - 1].outputs[i] - (i === target ? 1 : 0);
  }

  let deltas = outputDeltas;
  for (let l = layers.length - 1; l >= 0; l--) {
    const layer = layers[l];
    const { inputs, preActivations } = cache[l];
    const newDeltas = new Array(layer.inputSize).fill(0);

    for (let i = 0; i < layer.outputSize; i++) {
      let delta;
      if (layer.softmax) {
        delta = deltas[i];
      } else {
        delta = deltas[i] * reluDeriv(preActivations[i]);
      }

      const n = layer.neurons[i];
      for (let j = 0; j < layer.inputSize; j++) {
        newDeltas[j] += n.weights[j] * delta;
        n.weights[j] -= lr * delta * inputs[j];
      }
      n.bias -= lr * delta;
    }
    deltas = newDeltas;
  }
}

// Train on a simple task: input [1,0,0] -> class 0, [0,1,0] -> class 1
const trainingData = [
  { input: [1, 0, 0], label: 0 },
  { input: [0, 1, 0], label: 1 },
  { input: [1, 0, 0], label: 0 },
  { input: [0, 1, 0], label: 1 },
];

const lr = 0.1;

// Check initial forward
console.log("=== Initial forward ===");
for (const d of trainingData) {
  const { output } = forward(d.input);
  console.log(`Input: [${d.input}] -> Output: [${output.map(v => v.toFixed(4))}] NaN: ${output.some(v => !Number.isFinite(v))}`);
}

// Train 100 epochs
for (let epoch = 0; epoch < 100; epoch++) {
  let totalLoss = 0;
  let correct = 0;
  for (const d of trainingData) {
    const { output, cache } = forward(d.input);
    const loss = -Math.log(Math.max(output[d.label], 1e-15));
    totalLoss += loss;
    const predicted = output.indexOf(Math.max(...output));
    if (predicted === d.label) correct++;
    backward(d.label, lr, cache);
  }
  if (epoch % 20 === 0) {
    console.log(`Epoch ${epoch}: loss=${(totalLoss / trainingData.length).toFixed(4)} acc=${correct}/${trainingData.length}`);
  }
}

// Check final forward
console.log("\n=== Final forward ===");
for (const d of trainingData) {
  const { output } = forward(d.input);
  const predicted = output.indexOf(Math.max(...output));
  console.log(`Input: [${d.input}] -> Output: [${output.map(v => v.toFixed(4))}] predicted=${predicted} target=${d.label} NaN: ${output.some(v => !Number.isFinite(v))}`);
}

// Now test with 784 inputs -> 16 -> 16 -> 10
console.log("\n=== MNIST-sized test ===");
const mnistLayers = [
  { neurons: [], inputSize: 784, outputSize: 16 },
  { neurons: [], inputSize: 16, outputSize: 16 },
  { neurons: [], inputSize: 16, outputSize: 10, softmax: true },
];
for (const layer of mnistLayers) {
  for (let i = 0; i < layer.outputSize; i++) {
    layer.neurons.push({ weights: heInit(layer.inputSize), bias: 0 });
  }
}

function mnistForward(input) {
  let current = input;
  const cache = [];
  for (const layer of mnistLayers) {
    const preActivations = [];
    for (let i = 0; i < layer.outputSize; i++) {
      const n = layer.neurons[i];
      let sum = n.bias;
      for (let j = 0; j < layer.inputSize; j++) {
        sum += n.weights[j] * current[j];
      }
      preActivations.push(sum);
    }
    let outputs;
    if (layer.softmax) {
      outputs = softmax(preActivations);
    } else {
      outputs = preActivations.map(relu);
    }
    cache.push({ inputs: current, preActivations, outputs });
    current = outputs;
  }
  return { output: current, cache };
}

function mnistBackward(target, lr, cache) {
  const L = mnistLayers.length;
  const outputLayer = mnistLayers[L - 1];
  const outputDeltas = new Array(outputLayer.outputSize);
  for (let i = 0; i < outputLayer.outputSize; i++) {
    outputDeltas[i] = cache[L - 1].outputs[i] - (i === target ? 1 : 0);
  }

  let deltas = outputDeltas;
  for (let l = L - 1; l >= 0; l--) {
    const layer = mnistLayers[l];
    const { inputs, preActivations } = cache[l];
    const newDeltas = new Array(layer.inputSize).fill(0);

    for (let i = 0; i < layer.outputSize; i++) {
      let delta;
      if (layer.softmax) {
        delta = deltas[i];
      } else {
        delta = deltas[i] * reluDeriv(preActivations[i]);
      }

      const n = layer.neurons[i];
      for (let j = 0; j < layer.inputSize; j++) {
        newDeltas[j] += n.weights[j] * delta;
        n.weights[j] -= lr * delta * inputs[j];
      }
      n.bias -= lr * delta;
    }
    deltas = newDeltas;
  }
}

// Create 10 fake MNIST samples
const mnistData = [];
for (let s = 0; s < 10; s++) {
  const input = new Array(784).fill(0).map(() => Math.random());
  mnistData.push({ input, label: s });
}

// Forward check
console.log("Initial forward:");
const initResult = mnistForward(mnistData[0].input);
console.log(`  NaN in output: ${initResult.output.some(v => !Number.isFinite(v))}`);
console.log(`  NaN in weights: ${mnistLayers.some(l => l.neurons.some(n => n.weights.some(w => !Number.isFinite(w))))}`);
console.log(`  Output: [${initResult.output.map(v => v.toFixed(4))}]`);

// Train 50 epochs on all 10 samples
for (let epoch = 0; epoch < 50; epoch++) {
  let correct = 0;
  for (const d of mnistData) {
    const { output, cache } = mnistForward(d.input);
    const predicted = output.indexOf(Math.max(...output));
    if (predicted === d.label) correct++;
    mnistBackward(d.label, 0.1, cache);
  }
  if (epoch % 10 === 0) {
    const hasNaN = mnistLayers.some(l => l.neurons.some(n => n.weights.some(w => !Number.isFinite(w))));
    console.log(`Epoch ${epoch}: acc=${correct}/10 hasNaN=${hasNaN}`);
  }
}

// Final check
console.log("\nFinal forward:");
for (const d of mnistData) {
  const { output } = mnistForward(d.input);
  const predicted = output.indexOf(Math.max(...output));
  console.log(`  target=${d.label} predicted=${predicted} probs=[${output.map(v => v.toFixed(3))}] NaN=${output.some(v => !Number.isFinite(v))}`);
}

const hasNaN = mnistLayers.some(l => l.neurons.some(n => n.weights.some(w => !Number.isFinite(w))));
console.log(`\nFinal NaN in weights: ${hasNaN}`);
