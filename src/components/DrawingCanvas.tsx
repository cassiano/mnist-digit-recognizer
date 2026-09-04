/**
 * DrawingCanvas component — provides a canvas for users to draw digits.
 *
 * The canvas renders white strokes on a black background (matching MNIST format)
 * at 280×280 resolution. When "Recognize" is clicked, the canvas content is
 * captured as ImageData and sent to the parent for preprocessing and inference.
 *
 * Drawing uses mouse events (mousedown/move/up) with a thick brush (28px)
 * to produce bold strokes that downsample well to 28×28.
 */
import { useRef, useState, useEffect } from 'react'
import { CANVAS_SIZE, CANVAS_DOT_RADIUS, CANVAS_LINE_WIDTH } from '../constants'

interface DrawingCanvasProps {
  onRecognize: (imageData: ImageData) => void
  disabled?: boolean
}

export function DrawingCanvas({ onRecognize, disabled = false }: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [hasContent, setHasContent] = useState(false)
  /** Tracks the last mouse position to draw continuous lines between frames */
  const lastPoint = useRef<{ x: number; y: number } | null>(null)

  /** Initialize canvas with black background on mount */
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }, [])

  /**
   * Converts browser mouse coordinates to canvas pixel coordinates.
   * Accounts for CSS scaling by using getBoundingClientRect().
   */
  const getCanvasPoint = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return null
    const rect = canvas.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    }
  }

  /** Start drawing: record initial point and draw a dot */
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled) return
    setIsDrawing(true)
    setHasContent(true)
    const point = getCanvasPoint(e)
    if (point) {
      lastPoint.current = point
      const ctx = canvasRef.current?.getContext('2d')
      if (ctx) {
        ctx.beginPath()
        ctx.arc(point.x, point.y, CANVAS_DOT_RADIUS, 0, Math.PI * 2)
        ctx.fillStyle = '#fff'
        ctx.fill()
      }
    }
  }

  /** Continue drawing: draw a thick line from last point to current point */
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || disabled) return
    const point = getCanvasPoint(e)
    if (point && lastPoint.current) {
      const ctx = canvasRef.current?.getContext('2d')
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

  /** Clear canvas to black background */
  const handleClear = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      setHasContent(false)
    }
  }

  /** Capture canvas content and send to parent for recognition */
  const handleRecognize = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    onRecognize(imageData)
  }

  return (
    <div className="drawing-canvas">
      <h3>Draw a Digit</h3>
      <canvas
        ref={canvasRef}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          border: '2px solid var(--border)',
          borderRadius: '8px',
          cursor: disabled ? 'not-allowed' : 'crosshair',
          background: '#000',
        }}
      />
      <div className="canvas-controls">
        <button onClick={handleClear} disabled={disabled}>Clear</button>
        <button onClick={handleRecognize} disabled={disabled || !hasContent} className="primary">Recognize</button>
      </div>
    </div>
  )
}
