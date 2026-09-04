/**
 * TrainingPanel component — manages MNIST training workflow.
 *
 * Provides UI for configuring training parameters (epochs, learning rate, batch size)
 * and controls the training process. Implements an async, chunked training loop that
 * processes samples in ~16ms batches to keep the UI responsive.
 *
 * Training flow:
 *   1. Load MNIST dataset via MnistLoader (downloads gzipped binary files)
 *   2. Shuffle training indices each epoch (Fisher-Yates)
 *   3. Process samples in chunks, yielding to the event loop between chunks
 *   4. After each epoch, evaluate on test set and record accuracy
 *   5. Auto-save model to localStorage after training completes
 *
 * Supports aborting training mid-process and preserves partial results.
 */
import { useState, useRef, useEffect } from 'react'
import { Network } from '../neural-network/Network'
import { MnistLoader } from '../mnist/MnistLoader'
import type { TrainingResult } from '../neural-network/types'
import { timesMap } from '../utils'

/** localStorage key for persisting the trained model */
const STORAGE_KEY = 'mnist-nn-model-v2'
/** localStorage key for persisting training results */
const RESULTS_KEY = 'mnist-nn-results-v2'

interface TrainingPanelProps {
  network: Network
  onTrained: () => void
  onModelSaved: () => void
  onTrainingTick?: () => void
  initialResults?: TrainingResult[]
}

export function TrainingPanel({
  network,
  onTrained,
  onModelSaved,
  onTrainingTick,
  initialResults,
}: TrainingPanelProps) {
  const [status, setStatus] = useState('Ready to train')
  const [epochs, setEpochs] = useState(5)
  const [learningRate, setLearningRate] = useState(0.01)
  const [batchSize, setBatchSize] = useState(64)
  const [isTraining, setIsTraining] = useState(false)
  const [results, setResults] = useState<TrainingResult[]>(initialResults ?? [])

  // Refs to access current values in async callbacks without stale closures
  const networkRef = useRef(network)
  const mnistRef = useRef<MnistLoader | null>(null)
  const abortRef = useRef(false)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const onTrainedRef = useRef(onTrained)
  const onModelSavedRef = useRef(onModelSaved)
  const onTrainingTickRef = useRef(onTrainingTick)

  // Direct DOM refs for progress bar updates (avoids React re-render overhead)
  const progressFillRef = useRef<HTMLDivElement>(null)
  const epochLabelRef = useRef<HTMLSpanElement>(null)
  const percentLabelRef = useRef<HTMLSpanElement>(null)
  const samplesLabelRef = useRef<HTMLSpanElement>(null)
  const timeLabelRef = useRef<HTMLSpanElement>(null)

  /** Keep callback refs in sync with latest props */
  useEffect(() => {
    onTrainedRef.current = onTrained
    onModelSavedRef.current = onModelSaved
    onTrainingTickRef.current = onTrainingTick
  })

  /** Create a fresh MnistLoader instance on mount */
  useEffect(() => {
    mnistRef.current = new MnistLoader()
  }, [])

  /** Sync network ref when network prop changes (e.g., after architecture change) */
  useEffect(() => {
    networkRef.current = network
  }, [network])

  /**
   * Updates progress bar and labels via direct DOM manipulation.
   * This bypasses React state to avoid re-render overhead during training.
   */
  const updateDOM = (
    pct: number,
    epoch: number,
    totalEpochs: number,
    processed: number,
    total: number,
  ) => {
    if (progressFillRef.current) progressFillRef.current.style.width = `${pct}%`
    if (percentLabelRef.current)
      percentLabelRef.current.textContent = `${Math.round(pct)}%`
    if (epochLabelRef.current)
      epochLabelRef.current.textContent = `Epoch ${epoch} / ${totalEpochs}`
    if (samplesLabelRef.current)
      samplesLabelRef.current.textContent = `${processed.toLocaleString()} / ${total.toLocaleString()} samples`
  }

  /** Saves the current network weights/biases and training results to localStorage */
  const saveModel = (epochResults: TrainingResult[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, networkRef.current.serialize())
      localStorage.setItem(RESULTS_KEY, JSON.stringify(epochResults))
      onModelSavedRef.current()
    } catch {
      setStatus('Error: could not save model')
    }
  }

  /**
   * Main training handler — implements async, chunked training.
   *
   * Uses setTimeout-based chunking to process ~16ms of work per frame,
   * preventing the UI from freezing during the 60,000-sample training loop.
   * Each chunk processes as many samples as fit in 16ms (one frame at 60fps).
   */
  const handleTrain = async () => {
    if (isTraining || !mnistRef.current) return

    const net = networkRef.current

    setIsTraining(true)
    setResults([])

    abortRef.current = false

    // Start elapsed time counter
    let elapsed = 0

    timerRef.current = setInterval(() => {
      elapsed++

      if (timeLabelRef.current) {
        const mins = Math.floor(elapsed / 60)
        const secs = elapsed % 60

        timeLabelRef.current.textContent = `${mins}:${secs.toString().padStart(2, '0')}`
      }
    }, 1000)

    try {
      // Load MNIST dataset (downloads and decompresses gzipped binary files)
      setStatus('Loading MNIST dataset...')

      await mnistRef.current.load(setStatus)

      if (abortRef.current) {
        setStatus('Training aborted')
        setIsTraining(false)

        if (timerRef.current) clearInterval(timerRef.current)

        return
      }

      const trainingData = mnistRef.current.getTrainingData()
      const testData = mnistRef.current.getTestData()
      const total = epochs * trainingData.inputs.length

      setStatus('Training network...')

      await new Promise(r => setTimeout(r, 0)) // yield to render loading message

      updateDOM(0, 0, epochs, 0, total)

      const epochResults: TrainingResult[] = []
      let currentLR = learningRate

      for (let epoch = 0; epoch < epochs; epoch++) {
        if (abortRef.current) break

        let correct = 0

        // Fisher-Yates shuffle for random sampling each epoch
        const indices = timesMap(trainingData.inputs.length, i => i)
        for (let k = indices.length - 1; k > 0; k--) {
          const j = Math.floor(Math.random() * (k + 1))

          ;[indices[k], indices[j]] = [indices[j], indices[k]]
        }

        let i = 0
        const aborted = await new Promise<boolean>(resolve => {
          const chunk = () => {
            if (abortRef.current) {
              resolve(true)
              return
            }

            const batchSize = document.hidden ? 2000 : 100
            const end = Math.min(i + batchSize, indices.length)

            while (i < end) {
              if (abortRef.current) {
                resolve(true)
                return
              }

              const idx = indices[i]
              const output = net.forward(trainingData.inputs[idx])
              const predicted = output.indexOf(Math.max(...output))

              if (predicted === trainingData.labels[idx]) correct++

              net.backward(trainingData.labels[idx], currentLR)
              i++
            }
            if (!document.hidden) {
              const processed = epoch * indices.length + i

              updateDOM(
                (processed / total) * 100,
                epoch + 1,
                epochs,
                processed,
                total,
              )

              onTrainingTickRef.current?.()
            }
            if (i < indices.length) {
              setTimeout(chunk, 0)
            } else {
              if (document.hidden) {
                updateDOM(
                  ((epoch + 1) / epochs) * 100,
                  epoch + 1,
                  epochs,
                  (epoch + 1) * indices.length,
                  total,
                )
                onTrainingTickRef.current?.()
              }

              resolve(false)
            }
          }

          setTimeout(chunk, 0)
        })

        if (aborted) break

        // Evaluate on test set after each epoch
        let testCorrect = 0

        for (let ti = 0; ti < testData.inputs.length; ti++) {
          const output = net.forward(testData.inputs[ti])
          const predicted = output.indexOf(Math.max(...output))

          if (predicted === testData.labels[ti]) testCorrect++
        }

        const testAccuracy = testCorrect / testData.inputs.length
        const result: TrainingResult = {
          epoch: epoch + 1,
          loss: 0,
          accuracy: testAccuracy,
        }

        epochResults.push(result)
        setResults([...epochResults])
      }

      saveModel(epochResults)

      if (!abortRef.current) {
        setStatus('Training complete! Model saved.')
        onTrainedRef.current()
      } else {
        setStatus('Training aborted. Partial model saved.')
        onTrainedRef.current()
      }
    } catch (error) {
      setStatus(
        `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      )
    } finally {
      setIsTraining(false)

      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }

  /** Signals the training loop to stop at the next opportunity */
  const handleAbort = () => {
    abortRef.current = true
  }

  return (
    <div className="training-panel">
      <h3>Training Configuration</h3>

      <div className="training-config">
        <label>
          Epochs:
          <input
            type="number"
            value={epochs}
            onChange={e => setEpochs(parseInt(e.target.value) || 1)}
            min={1}
            max={100}
            disabled={isTraining}
          />
        </label>
        <label>
          Learning Rate:
          <input
            type="number"
            value={learningRate}
            onChange={e => setLearningRate(parseFloat(e.target.value) || 0.01)}
            min={0.001}
            max={1}
            step={0.01}
            disabled={isTraining}
          />
        </label>
        <label>
          Batch Size:
          <input
            type="number"
            value={batchSize}
            onChange={e => setBatchSize(parseInt(e.target.value) || 1)}
            min={1}
            max={128}
            disabled={isTraining}
          />
        </label>
      </div>

      <div className="training-controls">
        {!isTraining ? (
          <button onClick={handleTrain} className="primary">
            Start Training
          </button>
        ) : (
          <button onClick={handleAbort} className="danger">
            Abort
          </button>
        )}
      </div>

      <div className="training-status">
        {isTraining && (
          <div className="progress-section">
            <div className="progress-header">
              <span className="epoch-label" ref={epochLabelRef}>
                Epoch 0 / {epochs}
              </span>
              <span className="percent-label" ref={percentLabelRef}>
                0%
              </span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                ref={progressFillRef}
                style={{ width: '0%' }}
              />
            </div>
            <div className="progress-details">
              <span ref={samplesLabelRef}>0 / 0 samples</span>
              <span ref={timeLabelRef}>0:00</span>
            </div>
          </div>
        )}
        {!isTraining && <p>{status}</p>}
        {results.length > 0 && results[results.length - 1].accuracy > 0 && (
          <p className="accuracy">
            Test Accuracy:{' '}
            {(results[results.length - 1].accuracy * 100).toFixed(2)}%
          </p>
        )}
      </div>

      {results.length > 0 && (
        <div className="training-results">
          <h4>Epoch Results</h4>
          <div className="results-list">
            {results.map(r => (
              <div key={r.epoch} className="result-item">
                <span>Epoch {r.epoch}</span>
                <span>Loss: {r.loss.toFixed(4)}</span>
                <span>Acc: {(r.accuracy * 100).toFixed(2)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
