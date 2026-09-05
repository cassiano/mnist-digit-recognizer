/**
 * DrawingCanvas component — provides a canvas for users to draw digits.
 *
 * Uses two stacked canvases:
 * - Bottom: drawing canvas (white strokes on black, 280×280)
 * - Top: grid overlay (28×28 gray lines, transparent background)
 *
 * The grid is only a visual hint — it is never captured in the image data.
 * When "Recognize" is clicked, only the drawing canvas content is sent
 * to the parent for preprocessing and inference.
 *
 * Drawing uses mouse events (mousedown/move/up) with a thick brush (28px)
 * to produce bold strokes that downsample well to 28×28.
 */
import { useRef, useState, useEffect } from 'react'
import {
  CANVAS_SIZE,
  CANVAS_DOT_RADIUS,
  CANVAS_LINE_WIDTH,
  MNIST_IMAGE_COLS,
  MNIST_IMAGE_ROWS,
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
  /** Tracks the last mouse position to draw continuous lines between frames */
  const lastPoint = useRef<{ x: number; y: number } | null>(null)

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

  /** Initialize drawing canvas with black background and grid overlay on mount */
  useEffect(() => {
    const ctx = drawCanvasRef.current?.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    drawGrid()
  }, [])

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
