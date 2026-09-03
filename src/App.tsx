/**
 * App component — root component for the MNIST Digit Recognizer.
 *
 * Manages the application state including:
 *   - Neural network instance (with architecture configuration)
 *   - Training status and model persistence
 *   - Digit recognition from canvas input
 *
 * Layout uses CSS grid with 3 columns:
 *   Col 1: Architecture configuration
 *   Col 2: Network info + Training panel
 *   Col 3 (spans all rows): Drawing canvas + Status + Prediction result
 *   Cols 1-2 (row 2): Network visualization
 *
 * Model persistence: network weights are serialized to localStorage on training
 * completion and restored on page reload. The architecture is derived from the
 * saved layer sizes.
 */
import { useState, useRef } from 'react'
import { Network } from './neural-network/Network'
import { MnistLoader } from './mnist/MnistLoader'
import { DrawingCanvas } from './components/DrawingCanvas'
import { TrainingPanel } from './components/TrainingPanel'
import { NetworkInfo } from './components/NetworkInfo'
import { NetworkVisualization } from './components/NetworkVisualization'
import { PredictionResult } from './components/PredictionResult'
import type { Prediction } from './neural-network/types'

/** localStorage key for persisting the trained model */
const STORAGE_KEY = 'mnist-nn-model-v2'

const DEFAULT_HIDDEN_LAYER_SIZE = 16

/**
 * Creates a new network with the given hidden layer sizes.
 * Architecture: [784, ...hiddenLayerSizes, 10]
 * - Input: 784 (28×28 pixels)
 * - Output: 10 (digits 0-9)
 * - Default learning rate: 0.01, activation: ReLU
 */
function buildNetwork(hiddenLayerSizes: number[]) {
  const layers = [784, ...hiddenLayerSizes, 10]
  return new Network({ layers, learningRate: 0.01, activation: 'relu' })
}

/**
 * Attempts to load a previously saved model from localStorage.
 *
 * Validates the saved data by checking that at least one weight is non-zero
 * (all-zero weights indicate a failed or untrained model).
 *
 * @returns The restored network and its hidden layer sizes, or null if unavailable/invalid.
 */
function loadSavedState(): {
  network: Network
  hiddenLayerSizes: number[]
} | null {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (!saved) return null

    const net = Network.deserialize(saved)

    // Verify weights are non-zero (all-zero = untrained or failed)
    let hasNonZero = false

    for (const layer of net.layers) {
      for (const neuron of layer.neurons) {
        for (const w of neuron.weights) {
          if (w !== 0) {
            hasNonZero = true
            break
          }
        }

        if (hasNonZero) break
      }

      if (hasNonZero) break
    }

    if (!hasNonZero) {
      localStorage.removeItem(STORAGE_KEY)
      return null
    }

    // Extract hidden layer sizes (exclude input 784 and output 10)
    const hiddenLayerSizes = net.layers.map(l => l.outputSize).slice(0, -1)

    return { network: net, hiddenLayerSizes }
  } catch {
    localStorage.removeItem(STORAGE_KEY)

    return null
  }
}

/** Shared MNIST loader instance — avoids re-downloading on every recognition */
const sharedMnistLoader = new MnistLoader()

function App() {
  /** Initialize state from localStorage or defaults */
  const [initData] = useState(() => {
    const saved = loadSavedState()

    if (saved) {
      return {
        sizes: saved.hiddenLayerSizes,
        network: saved.network,
        trained: true,
      }
    }

    const sizes = [DEFAULT_HIDDEN_LAYER_SIZE, DEFAULT_HIDDEN_LAYER_SIZE]

    return { sizes, network: buildNetwork(sizes), trained: false }
  })

  const [hiddenLayerSizes, setHiddenLayerSizes] = useState(initData.sizes)
  const [network, setNetwork] = useState(initData.network)
  /** Ref to access current network in async callbacks without stale closures */
  const networkRef = useRef(network)
  const [isTrained, setIsTrained] = useState(initData.trained)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [trainingTick, setTrainingTick] = useState(0)
  const [status, setStatus] = useState(
    initData.trained
      ? 'Model loaded from previous session.'
      : 'Network initialized. Please train the network first.',
  )

  // Keep ref in sync with state for use in handleRecognize
  networkRef.current = network

  const layerDescription = `784 → ${hiddenLayerSizes.join(' → ')} → 10`

  /** Called by TrainingPanel after training completes */
  const handleTrained = () => {
    setIsTrained(true)
    setStatus('Network trained! Draw a digit to recognize.')
  }

  /** Called by TrainingPanel after model is saved to localStorage */
  const handleModelSaved = () => {
    setStatus('Model saved to local storage.')
  }

  /** Called periodically during training to update the network visualization */
  const handleTrainingTick = () => {
    setTrainingTick(t => t + 1)
  }

  /**
   * Handles digit recognition from the drawing canvas.
   *
   * 1. Preprocesses canvas ImageData to 28×28 grayscale (784 values)
   * 2. Runs forward pass through the trained network
   * 3. Displays the predicted digit and confidence
   *
   * Uses networkRef.current to avoid stale closure issues with React state.
   */
  const handleRecognize = (imageData: ImageData) => {
    if (!isTrained) {
      setStatus('Please train the network first!')
      return
    }

    try {
      const inputs = sharedMnistLoader.preprocessCanvasData(imageData)
      const net = networkRef.current
      const result = net.predict(inputs)

      setPrediction(result)
      setStatus(
        `Recognized digit: ${result.digit} with ${(result.confidence * 100).toFixed(2)}% confidence`,
      )
    } catch (error) {
      setStatus(
        `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      )
    }
  }

  /** Adds a new hidden layer with the default size to the architecture */
  const addLayer = () => {
    const newSizes = [...hiddenLayerSizes, DEFAULT_HIDDEN_LAYER_SIZE]

    setHiddenLayerSizes(newSizes)
    setNetwork(buildNetwork(newSizes))
    setIsTrained(false)
  }

  /** Removes a hidden layer by index (minimum 1 layer required) */
  const removeLayer = (index: number) => {
    if (hiddenLayerSizes.length <= 1) return

    const newSizes = hiddenLayerSizes.filter((_, i) => i !== index)

    setHiddenLayerSizes(newSizes)
    setNetwork(buildNetwork(newSizes))
    setIsTrained(false)
  }

  /** Updates the neuron count for a specific hidden layer (clamped to 1-256) */
  const updateLayer = (index: number, size: number) => {
    const newSizes = [...hiddenLayerSizes]

    newSizes[index] = Math.max(1, Math.min(256, size || 1))

    setHiddenLayerSizes(newSizes)
    setNetwork(buildNetwork(newSizes))
    setIsTrained(false)
  }

  /** Resets the model: clears localStorage and creates a fresh untrained network */
  const handleReset = () => {
    localStorage.removeItem(STORAGE_KEY)

    const fresh = buildNetwork(hiddenLayerSizes)

    setNetwork(fresh)
    setIsTrained(false)
    setPrediction(null)
    setStatus('Model reset. Please train the network first.')
  }

  return (
    <div className="app">
      <header>
        <h1>MNIST Digit Recognizer</h1>
        <p>Neural Network with {layerDescription} architecture</p>
      </header>

      <main>
        {/* Column 1: Architecture configuration */}
        <div className="col col-arch">
          <div className="architecture-config">
            <h3>Architecture</h3>
            <div className="layers-list">
              <div className="layer-row input-row">
                <span className="layer-type">Input</span>
                <span className="layer-size">784</span>
              </div>
              {hiddenLayerSizes.map((size, index) => (
                <div key={index} className="layer-row">
                  <span className="layer-type">Hidden {index + 1}</span>
                  <input
                    type="number"
                    value={size}
                    onChange={e =>
                      updateLayer(index, parseInt(e.target.value) || 1)
                    }
                    min={1}
                    max={256}
                  />
                  <button
                    className="remove-btn"
                    onClick={() => removeLayer(index)}
                    disabled={hiddenLayerSizes.length <= 1}
                    title="Remove layer"
                  >
                    ×
                  </button>
                </div>
              ))}
              <div className="layer-row output-row">
                <span className="layer-type">Output</span>
                <span className="layer-size">10</span>
              </div>
            </div>
            <button className="add-layer-btn" onClick={addLayer}>
              + Add Hidden Layer
            </button>
          </div>
          {isTrained && (
            <button className="reset-btn" onClick={handleReset}>
              Reset Model
            </button>
          )}
        </div>

        {/* Column 2: Network info + Training configuration */}
        <div className="col col-training">
          <NetworkInfo network={network} />
          <TrainingPanel
            network={network}
            onTrained={handleTrained}
            onModelSaved={handleModelSaved}
            onTrainingTick={handleTrainingTick}
          />
        </div>

        {/* Column 3: Drawing canvas + status + prediction (spans all rows) */}
        <div className="col col-canvas center-panel">
          <DrawingCanvas onRecognize={handleRecognize} disabled={!isTrained} />
          <div className="status-bar">
            <p>{status}</p>
          </div>
          <PredictionResult prediction={prediction} />
        </div>

        {/* Cols 1-2, Row 2: Network visualization diagram */}
        <div className="col col-viz">
          <NetworkVisualization network={network} trainingTick={trainingTick} />
        </div>
      </main>
    </div>
  )
}

export default App
