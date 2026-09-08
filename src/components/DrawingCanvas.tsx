/**
 * DrawingCanvas component — provides a canvas for users to draw digits.
 *
 * Uses two stacked canvases:
 * - Bottom: drawing canvas (white strokes on black, 280×280)
 * - Top: grid overlay (28×28 gray lines, transparent background)
 *
 * A live 28×28 preview is shown in a side panel, displaying the exact
 * downsampled image the network will process.
 *
 * When "Recognize" is clicked, only the drawing canvas content is sent
 * to the parent for preprocessing and inference.
 *
 * Drawing uses mouse events (mousedown/move/up) with a thick brush (28px)
 * to produce bold strokes that downsample well to 28×28.
 */
import { useRef, useState, useEffect, useCallback } from 'react'
import { MnistLoader } from '../mnist/MnistLoader'
import {
  CANVAS_SIZE,
  CANVAS_DOT_RADIUS,
  CANVAS_LINE_WIDTH,
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
  MNIST_PREVIEW_SIZE,
} from '../constants'

interface DrawingCanvasProps {
  onRecognize: (imageData: ImageData) => void
  disabled?: boolean
}

export function DrawingCanvas({
  onRecognize,
  disabled = false,
}: DrawingCanvasProps) {
  const [isDrawing, setIsDrawing] = useState(false)
  const [hasContent, setHasContent] = useState(false)

  const drawCanvasRef = useRef<HTMLCanvasElement>(null)
  const gridCanvasRef = useRef<HTMLCanvasElement>(null)
  const previewCanvasRef = useRef<HTMLCanvasElement>(null)
  /** Offscreen canvas used for reading drawing pixel data */
  const offscreenRef = useRef<HTMLCanvasElement | null>(null)
  const mnistLoaderRef = useRef(new MnistLoader())
  /** Tracks the last mouse position to draw continuous lines between frames */
  const lastPoint = useRef<{ x: number; y: number } | null>(null)
  /** requestAnimationFrame id for throttling preview updates */
  const rafId = useRef(0)

  /** Renders the 28×28 preview using the same preprocessing as the network */
  const updatePreview = useCallback(() => {
    const drawCanvas = drawCanvasRef.current
    const previewCanvas = previewCanvasRef.current
    if (!drawCanvas || !previewCanvas) return

    let offscreen = offscreenRef.current
    if (!offscreen) {
      offscreen = document.createElement('canvas')
      offscreen.width = CANVAS_SIZE
      offscreen.height = CANVAS_SIZE
      offscreenRef.current = offscreen
    }

    const offCtx = offscreen.getContext('2d')
    const previewCtx = previewCanvas.getContext('2d')
    if (!offCtx || !previewCtx) return

    // Capture the drawing canvas content
    offCtx.drawImage(drawCanvas, 0, 0)
    const imageData = offCtx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE)

    // Get the centered, normalized 28×28 image (same as network input)
    const image = mnistLoaderRef.current.preprocessCanvasData(imageData)

    // Convert [0,1] floats back to grayscale pixel data
    const pixelData = new ImageData(MNIST_IMAGE_COLS, MNIST_IMAGE_ROWS)

    for (let i = 0; i < image.length; i++) {
      const v = Math.round(image[i] * 255)
      const idx = i * 4
      pixelData.data[idx] = v
      pixelData.data[idx + 1] = v
      pixelData.data[idx + 2] = v
      pixelData.data[idx + 3] = 255
    }

    offCtx.putImageData(pixelData, 0, 0)

    // Scale 28×28 → display size with crisp pixel rendering
    previewCtx.imageSmoothingEnabled = false
    previewCtx.clearRect(0, 0, MNIST_PREVIEW_SIZE, MNIST_PREVIEW_SIZE)
    previewCtx.drawImage(
      offscreen,
      0, 0, MNIST_IMAGE_COLS, MNIST_IMAGE_ROWS,
      0, 0, MNIST_PREVIEW_SIZE, MNIST_PREVIEW_SIZE,
    )
  }, [])

  /** Throttled preview update via requestAnimationFrame */
  const schedulePreviewUpdate = useCallback(() => {
    cancelAnimationFrame(rafId.current)
    rafId.current = requestAnimationFrame(updatePreview)
  }, [updatePreview])

  /** Draws a 28×28 grid of gray lines on the overlay canvas */
  const drawGrid = () => {
    const ctx = gridCanvasRef.current?.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)

    const cellW = CANVAS_SIZE / MNIST_IMAGE_COLS
    const cellH = CANVAS_SIZE / MNIST_IMAGE_ROWS

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'
    ctx.lineWidth = 1

    for (let x = 1; x < MNIST_IMAGE_COLS; x++) {
      ctx.beginPath()
      ctx.moveTo(Math.round(x * cellW) + 0.5, 0)
      ctx.lineTo(Math.round(x * cellW) + 0.5, CANVAS_SIZE)
      ctx.stroke()
    }

    for (let y = 1; y < MNIST_IMAGE_ROWS; y++) {
      ctx.beginPath()
      ctx.moveTo(0, Math.round(y * cellH) + 0.5)
      ctx.lineTo(CANVAS_SIZE, Math.round(y * cellH) + 0.5)
      ctx.stroke()
    }
  }

  /** Initialize drawing canvas, grid overlay, and preview on mount */
  useEffect(() => {
    const ctx = drawCanvasRef.current?.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    drawGrid()
    updatePreview()

    return () => cancelAnimationFrame(rafId.current)
  }, [updatePreview])

  /**
   * Converts browser mouse coordinates to canvas pixel coordinates.
   * Accounts for CSS scaling by using getBoundingClientRect().
   */
  const getCanvasPoint = (e: React.MouseEvent<HTMLDivElement>) => {
    const canvas = drawCanvasRef.current
    if (!canvas) return null

    const rect = canvas.getBoundingClientRect()

    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    }
  }

  /** Start drawing: record initial point and draw a dot */
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (disabled) return

    setIsDrawing(true)
    setHasContent(true)

    const point = getCanvasPoint(e)

    if (point) {
      lastPoint.current = point

      const ctx = drawCanvasRef.current?.getContext('2d')

      if (ctx) {
        ctx.beginPath()
        ctx.arc(point.x, point.y, CANVAS_DOT_RADIUS, 0, Math.PI * 2)
        ctx.fillStyle = '#fff'
        ctx.fill()
      }

      schedulePreviewUpdate()
    }
  }

  /** Continue drawing: draw a thick line from last point to current point */
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawing || disabled) return

    const point = getCanvasPoint(e)

    if (point && lastPoint.current) {
      const ctx = drawCanvasRef.current?.getContext('2d')

      if (ctx) {
        ctx.beginPath()
        ctx.moveTo(lastPoint.current.x, lastPoint.current.y)
        ctx.lineTo(point.x, point.y)
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = CANVAS_LINE_WIDTH
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.stroke()
      }

      lastPoint.current = point
      schedulePreviewUpdate()
    }
  }

  /** Stop drawing */
  const handleMouseUp = () => {
    setIsDrawing(false)
    lastPoint.current = null
  }

  /** Clear drawing canvas to black background */
  const handleClear = () => {
    const ctx = drawCanvasRef.current?.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    setHasContent(false)
    updatePreview()
  }

  /** Capture drawing canvas content (without grid) and send to parent */
  const handleRecognize = () => {
    const canvas = drawCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    onRecognize(imageData)
  }

  const canvasStyle = {
    border: '2px solid var(--border)',
    borderRadius: '8px',
    background: '#000',
  }

  return (
    <div className="drawing-canvas">
      <h3>Draw a Digit</h3>
      <div className="canvas-and-preview">
        <div className="canvas-column">
          <span className="canvas-dim-label">
            {CANVAS_SIZE}×{CANVAS_SIZE}
          </span>
          <div
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{
              position: 'relative',
              width: CANVAS_SIZE,
              height: CANVAS_SIZE,
              cursor: disabled ? 'not-allowed' : 'crosshair',
            }}
          >
            <canvas
              ref={drawCanvasRef}
              width={CANVAS_SIZE}
              height={CANVAS_SIZE}
              style={canvasStyle}
            />
            <canvas
              ref={gridCanvasRef}
              width={CANVAS_SIZE}
              height={CANVAS_SIZE}
              style={{
                ...canvasStyle,
                background: 'transparent',
                position: 'absolute',
                top: 0,
                left: 0,
                pointerEvents: 'none',
              }}
            />
          </div>
        </div>
        <div className="preview-section">
          <span className="canvas-dim-label">
            {MNIST_IMAGE_COLS}×{MNIST_IMAGE_ROWS}
          </span>
          <canvas
            ref={previewCanvasRef}
            width={MNIST_PREVIEW_SIZE}
            height={MNIST_PREVIEW_SIZE}
            style={{
              width: MNIST_PREVIEW_SIZE,
              height: MNIST_PREVIEW_SIZE,
              border: '2px solid var(--border)',
              borderRadius: '8px',
              background: '#000',
              imageRendering: 'pixelated',
            }}
          />
        </div>
      </div>
      <div className="canvas-controls">
        <button onClick={handleClear} disabled={disabled}>
          Clear
        </button>
        <button
          onClick={handleRecognize}
          disabled={disabled || !hasContent}
          className="primary"
        >
          Recognize
        </button>
      </div>
    </div>
  )
}
