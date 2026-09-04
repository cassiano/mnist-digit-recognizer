// Neural Network Architecture
export const INPUT_SIZE = 784 // 28x28 pixels
export const OUTPUT_SIZE = 10 // digits 0-9
export const DEFAULT_HIDDEN_LAYER_SIZE = 16
export const MIN_LAYER_SIZE = 1
export const MAX_LAYER_SIZE = 256
export const MIN_HIDDEN_LAYERS = 1

// Training Defaults
export const DEFAULT_LEARNING_RATE = 0.01
export const DEFAULT_EPOCHS = 5
export const DEFAULT_BATCH_SIZE = 64
export const LEARNING_RATE_DECAY = 0.999

// Numerical Stability
export const EPSILON = 1e-15

// MNIST Dataset
export const MNIST_IMAGE_ROWS = 28
export const MNIST_IMAGE_COLS = 28
export const MNIST_IMAGE_MAGIC = 2051
export const MNIST_LABEL_MAGIC = 2049
export const MNIST_PIXEL_MAX = 255

// Drawing Canvas
export const CANVAS_SIZE = 280
export const CANVAS_DOT_RADIUS = 18
export const CANVAS_LINE_WIDTH = 28

// Training Performance
export const FOREGROUND_BATCH_SIZE = 100
export const BACKGROUND_BATCH_SIZE = 2000
export const TIMER_INTERVAL_MS = 1000

// Network Visualization
export const SVG_WIDTH = 1100
export const SVG_HEIGHT = 480
export const MAX_DISPLAY_NEURONS = 8
export const SVG_PADDING_Y = 50
export const LAYER_HEIGHT_OFFSET = 30
export const MIN_LAYER_HEIGHT = 60
export const NEURON_SPACING = 28
export const NEURON_RADIUS_LARGE = 14
export const NEURON_RADIUS_SMALL = 10
export const NEURON_ACTIVE_THRESHOLD = 0.1

// He Initialization
export const HE_INIT_FACTOR = 2

// Storage Keys
export const STORAGE_KEY_MODEL = 'mnist-nn-model-v2'
export const STORAGE_KEY_RESULTS = 'mnist-nn-results-v2'
export const STORAGE_KEY_THEME = 'mnist-nn-theme'
