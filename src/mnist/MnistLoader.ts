import type { TrainingData } from '../neural-network/types'
import { map } from '../utils'

const MNIST_IMAGE_DIMENSIONS = { rows: 28, cols: 28 }

// Files in Big Endian IDX format, gzipped.
const MNIST_URLS = {
  trainImages: 'mnist/train-images-idx3-ubyte',
  trainLabels: 'mnist/train-labels-idx1-ubyte',
  testImages: 'mnist/t10k-images-idx3-ubyte',
  testLabels: 'mnist/t10k-labels-idx1-ubyte',
}
// const MNIST_URLS = {
//   trainImages:
//     'http://localhost:5173/mnist-digit-recognizer/mnist/train-images-idx3-ubyte',
//   trainLabels:
//     'http://localhost:5173/mnist-digit-recognizer/mnist/train-labels-idx1-ubyte',
//   testImages:
//     'http://localhost:5173/mnist-digit-recognizer/mnist/t10k-images-idx3-ubyte',
//   testLabels:
//     'http://localhost:5173/mnist-digit-recognizer/mnist/t10k-labels-idx1-ubyte',
// }

export class MnistLoader {
  private trainData: TrainingData | null = null
  private testData: TrainingData | null = null
  private loaded = false

  async load(onProgress?: (msg: string) => void): Promise<void> {
    if (this.loaded) return

    onProgress?.('Loading MNIST training images...')
    const trainImages = await this.downloadAndParseImages(
      MNIST_URLS.trainImages,
    )

    onProgress?.('Loading MNIST training labels...')
    const trainLabels = await this.downloadAndParseLabels(
      MNIST_URLS.trainLabels,
    )

    onProgress?.('Loading MNIST test images...')
    const testImages = await this.downloadAndParseImages(MNIST_URLS.testImages)

    onProgress?.('Loading MNIST test labels...')
    const testLabels = await this.downloadAndParseLabels(MNIST_URLS.testLabels)

    this.trainData = { inputs: trainImages, labels: trainLabels }
    this.testData = { inputs: testImages, labels: testLabels }
    this.loaded = true

    onProgress?.(
      `Loaded ${trainImages.length} training and ${testImages.length} test samples`,
    )
  }

  getTrainingData(): TrainingData {
    if (!this.trainData) throw new Error('Data not loaded. Call load() first.')

    return this.trainData
  }

  getTestData(): TrainingData {
    if (!this.testData) throw new Error('Data not loaded. Call load() first.')

    return this.testData
  }

  isLoaded(): boolean {
    return this.loaded
  }

  /**
   * Downloads and parses MNIST image data from a gzipped IDX binary file.
   *
   * File format (after decompression):
   *   Bytes 0-3:   Magic number (2051 for images)
   *   Bytes 4-7:   Number of images
   *   Bytes 8-11:  Number of rows per image (MNIST_IMAGE_DIMENSIONS.rows = 28)
   *   Bytes 12-15: Number of columns per image (MNIST_IMAGE_DIMENSIONS.cols = 28)
   *   Bytes 16+:   Pixel values (0-255), one byte per pixel, row-major order
   *
   * Each pixel is normalized to [0, 1] by dividing by 255.
   * Returns a 2D array where each inner array is a flattened 28×28 image (784 values).
   */
  private async downloadAndParseImages(url: string): Promise<number[][]> {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Failed to download ${url}`)

    const buffer = await response.arrayBuffer()
    const data = new Uint8Array(buffer)
    const decompressed = await this.gunzip(data)
    const view = new DataView(decompressed.buffer) // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView

    const magic = view.getUint32(0, false)
    if (magic !== 2051)
      throw new Error(`Invalid magic number for images: ${magic}`)

    const count = view.getUint32(4, false)
    const rows = view.getUint32(8, false)
    const cols = view.getUint32(12, false)

    const images: number[][] = []
    let offset = 16

    for (let i = 0; i < count; i++) {
      const image: number[] = []

      for (let j = 0; j < rows * cols; j++)
        image.push(decompressed[offset++] / 255)

      images.push(image)
    }

    return images
  }

  /**
   * Downloads and parses MNIST label data from a gzipped IDX binary file.
   *
   * File format (after decompression):
   *   Bytes 0-3: Magic number (2049 for labels)
   *   Bytes 4-7: Number of labels
   *   Bytes 8+:  Label values (0-9), one byte per label
   */
  private async downloadAndParseLabels(url: string): Promise<number[]> {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Failed to download ${url}`)

    const buffer = await response.arrayBuffer()
    const data = new Uint8Array(buffer)
    const decompressed = await this.gunzip(data)
    const view = new DataView(decompressed.buffer) // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/DataView

    const magic = view.getUint32(0, false)
    if (magic !== 2049)
      throw new Error(`Invalid magic number for labels: ${magic}`)

    const count = view.getUint32(4, false)
    const labels: number[] = []

    for (let i = 0; i < count; i++) labels.push(decompressed[8 + i])

    return labels
  }

  /**
   * Decompresses gzip data using the browser's built-in DecompressionStream API.
   *
   * This avoids needing external libraries (like pako) by leveraging the native
   * streaming decompression available in modern browsers.
   *
   * Process:
   *   1. Creates a DecompressionStream in 'gzip' mode
   *   2. Writes the compressed data to the writable side
   *   3. Reads decompressed chunks from the readable side
   *   4. Concatenates all chunks into a single Uint8Array
   */
  private async gunzip(data: Uint8Array): Promise<Uint8Array> {
    if (typeof DecompressionStream === 'undefined')
      throw new Error('DecompressionStream not supported in this browser')

    const ds = new DecompressionStream('gzip')
    const writer = ds.writable.getWriter()
    const reader = ds.readable.getReader()

    const chunks: Uint8Array[] = []
    let totalLength = 0

    const readTask = (async () => {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        chunks.push(value)
        totalLength += value.length
      }
    })()

    await writer.write(data.buffer as ArrayBuffer)
    await writer.close()
    await readTask

    const result = new Uint8Array(totalLength)
    let offset = 0

    for (const chunk of chunks) {
      result.set(chunk, offset)

      offset += chunk.length
    }

    return result
  }

  /**
   * Preprocesses canvas ImageData to match MNIST format.
   *
   * Downsamples the canvas (280×280) to 28×28 by direct pixel sampling,
   * then normalizes pixel values from [0, 255] to [0, 1].
   *
   * Returns a flat array of 784 values (28×28), ready for network input.
   * The canvas draws white digits on black background, matching MNIST's format.
   */
  preprocessCanvasData(imageData: ImageData): number[] {
    const inputs: number[] = []
    const { width, height, data } = imageData
    const stepX = width / MNIST_IMAGE_DIMENSIONS.cols
    const stepY = height / MNIST_IMAGE_DIMENSIONS.rows

    for (let y = 0; y < MNIST_IMAGE_DIMENSIONS.rows; y++) {
      for (let x = 0; x < MNIST_IMAGE_DIMENSIONS.cols; x++) {
        const srcX = Math.floor(x * stepX)
        const srcY = Math.floor(y * stepY)
        const idx = (srcY * width + srcX) * 4

        inputs.push(data[idx] / 255)
      }
    }

    return inputs
  }

  /**
   * Renders an MNIST image as a text-based visualization using Unicode block characters.
   *
   * Maps each pixel value (0-1) to a character from a gradient string,
   * where darker pixels (closer to 0) map to lighter characters and
   * brighter pixels (closer to 1) map to denser block characters.
   * Each character is repeated twice to maintain approximate square aspect ratio.
   *
   * @param type - 'trainData' for training images, 'testData' for test images
   * @param index - Index of the image within the selected dataset (0-based)
   * @returns A string containing the 28×28 image rendered as text with newline-separated rows
   * @throws If data has not been loaded (call `load()` first) or index is out of bounds
   */
  imageAsText(type: 'trainData' | 'testData', index: number): string {
    const dataset = this[type]
    if (!dataset) throw new Error('Data not loaded. Call load() first.')

    const image = dataset.inputs[index]
    if (!image) throw new Error(`Image index out of bounds: ${index}`)

    const gradient = ' ░▒▓▉█'

    let text = ''

    for (let row = 0; row < MNIST_IMAGE_DIMENSIONS.rows; row++) {
      for (let col = 0; col < MNIST_IMAGE_DIMENSIONS.cols; col++) {
        const pixelIndex = row * MNIST_IMAGE_DIMENSIONS.cols + col
        const value = image[pixelIndex]
        const charIndex = Math.trunc(
          map(value, 0, 1, 0, gradient.length - 1, true),
        )
        const unicodeChar = gradient[charIndex]

        text += unicodeChar.repeat(2)
      }

      text += '\n'
    }

    return text
  }
}
