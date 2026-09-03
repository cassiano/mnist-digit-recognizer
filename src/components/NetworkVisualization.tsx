/**
 * NetworkVisualization component — renders an SVG diagram of the neural network.
 *
 * Displays neurons as circles and weighted connections as colored lines.
 * - Line color: teal (#4ecdc4) for positive weights, red (#e94560) for negative
 * - Line thickness: proportional to weight magnitude
 * - Neuron fill: intensity based on activation value (brighter = more active)
 * - Neuron stroke: teal for active neurons (activation > 0.1), gray for inactive
 *
 * For large layers (>8 neurons), only the first and last few neurons are shown
 * with an ellipsis (⋮) in the middle to keep the diagram readable.
 *
 * SVG filters add a glow effect to active neurons and connections.
 */
import { useMemo } from 'react'
import type { Network } from '../neural-network/Network'

interface NetworkVisualizationProps {
  network: Network
  trainingTick?: number
}

export function NetworkVisualization({
  network,
  trainingTick,
}: NetworkVisualizationProps) {
  const svgWidth = 1100
  const svgHeight = 480
  const maxDisplayNeurons = 8
  const paddingY = 50

  const getLayerHeight = (size: number) => {
    const maxH = svgHeight - paddingY * 2 - 30

    if (size <= maxDisplayNeurons) return Math.max(60, size * 28)

    return maxH
  }

  const getLayerDisplay = (size: number, isInputLayer: boolean) => {
    if (!isInputLayer || size <= maxDisplayNeurons)
      return {
        neurons: Array.from({ length: size }, (_, i) => i),
        hasEllipsis: false,
      }
    const half = Math.floor(maxDisplayNeurons / 2)

    return {
      neurons: [
        ...Array.from({ length: half }, (_, i) => i),
        ...Array.from({ length: half }, (_, i) => size - half + i),
      ],
      hasEllipsis: true,
    }
  }

  const getNeuronY = (
    index: number,
    total: number,
    layerHeight: number,
  ): number => {
    const startY = (svgHeight - layerHeight) / 2
    if (total === 1) return startY + layerHeight / 2
    return startY + (index / (total - 1)) * layerHeight
  }

  const layerData = useMemo(() => {
    const layerSizes = [784, ...network.layers.map(l => l.outputSize)]
    const layerLabels = [
      'Input',
      ...network.layers.map((_l, i) =>
        i === network.layers.length - 1 ? 'Output' : `Hidden ${i + 1}`,
      ),
    ]
    const layerActivations = network.layers.map(layer => layer.activation)
    const layerSpacing = svgWidth / (layerSizes.length + 1)

    return layerSizes.map((size, l) => {
      const x = layerSpacing * (l + 1)
      const h = getLayerHeight(size)
      const display = getLayerDisplay(size, l === 0)
      const yPositions = display.neurons.map((_, i) =>
        getNeuronY(i, display.neurons.length, h),
      )

      const neurons: {
        x: number
        y: number
        activation: number
        origIndex: number
      }[] = []

      for (let i = 0; i < display.neurons.length; i++) {
        const origIdx = display.neurons[i]
        let activation = 0

        if (l > 0 && network.layers[l - 1].outputs.length > 0) {
          activation = network.layers[l - 1].outputs[origIdx] ?? 0
        } else if (l === 0) {
          activation = 1
        }

        neurons.push({ x, y: yPositions[i], activation, origIndex: origIdx })
      }

      return {
        x,
        h,
        size,
        display,
        neurons,
        label: layerLabels[l],
        activation: layerActivations[l - 1] ?? 'input',
      }
    })
    // trainingTick forces recalculation when network outputs change during training
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [network.layers, trainingTick])

  /**
   * Computes all connection lines between consecutive layers.
   * Each connection stores start/end coordinates, weight, and the maximum
   * absolute weight in the layer (used for normalizing visual intensity).
   */
  const connections: {
    x1: number
    y1: number
    x2: number
    y2: number
    weight: number
    maxAbsWeight: number
  }[] = useMemo(() => {
    const conns: typeof connections = []

    for (let l = 0; l < layerData.length - 1; l++) {
      const curr = layerData[l]
      const next = layerData[l + 1]

      // Find max absolute weight in this layer pair for normalization
      let maxAbs = 0

      for (const n of next.neurons) {
        for (const c of curr.neurons) {
          const w = network.layers[l].neurons[n.origIndex].weights[c.origIndex]

          if (Math.abs(w) > maxAbs) maxAbs = Math.abs(w)
        }
      }

      if (maxAbs === 0) maxAbs = 1

      // Create a connection line for each weight
      for (const n of next.neurons) {
        for (const c of curr.neurons) {
          const w = network.layers[l].neurons[n.origIndex].weights[c.origIndex]

          conns.push({
            x1: curr.x,
            y1: c.y,
            x2: next.x,
            y2: n.y,
            weight: w,
            maxAbsWeight: maxAbs,
          })
        }
      }
    }

    return conns
  }, [layerData, network.layers])

  return (
    <div className="network-visualization">
      <h3>Network Visualization</h3>
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="network-svg">
        <defs>
          {/* Glow filter for active neurons */}
          <filter id="neuron-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Subtle glow filter for connections */}
          <filter
            id="connection-glow"
            x="-50%"
            y="-50%"
            width="200%"
            height="200%"
          >
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Draw weighted connections between neurons */}
        {connections.map((conn, i) => {
          const normalizedWeight = conn.weight / conn.maxAbsWeight
          const absNorm = Math.abs(normalizedWeight)
          const opacity = 0.08 + absNorm * 0.55
          const strokeWidth = 0.6 + absNorm * 2.5
          const color = normalizedWeight >= 0 ? '#4ecdc4' : '#e94560'
          return (
            <line
              key={i}
              x1={conn.x1}
              y1={conn.y1}
              x2={conn.x2}
              y2={conn.y2}
              stroke={color}
              strokeWidth={strokeWidth}
              opacity={opacity}
            />
          )
        })}

        {/* Draw layers: background rect, neurons, labels */}
        {layerData.map((layer, l) => {
          const actLabel =
            l === 0 ? '' : layer.activation === 'softmax' ? 'softmax' : 'relu'

          return (
            <g key={l}>
              {/* Semi-transparent background rect for the layer */}
              <rect
                x={layer.x - 30}
                y={(svgHeight - layer.h) / 2 - 8}
                width={60}
                height={layer.h + 16}
                rx={14}
                fill={
                  l === 0 || l === layerData.length - 1 ? '#4ecdc4' : '#e94560'
                }
                opacity={0.08}
              />

              {/* Neuron circles with activation-based coloring */}
              {layer.neurons.map((neuron, i) => {
                const isActive = neuron.activation > 0.1
                const intensity = Math.min(1, neuron.activation)
                const baseColor =
                  l === 0 || l === layerData.length - 1 ? '#4ecdc4' : '#e94560'
                const fillColor = `color-mix(in srgb, ${baseColor} ${Math.round(30 + intensity * 70)}%, #f5f5f5)`
                const strokeColor = isActive ? baseColor : '#999'
                const r = layer.size <= 12 ? 14 : 10
                const isOutputLayer = l === layerData.length - 1

                return (
                  <g key={i}>
                    <circle
                      cx={neuron.x}
                      cy={neuron.y}
                      r={r}
                      fill={fillColor}
                      stroke={strokeColor}
                      strokeWidth={isActive ? 2 : 0.8}
                      opacity={0.7 + intensity * 0.3}
                      filter={isActive ? 'url(#neuron-glow)' : undefined}
                    />
                    {isOutputLayer && (
                      <text
                        x={neuron.x + r + 6}
                        y={neuron.y + 4}
                        fill="#333"
                        fontSize="11"
                        fontFamily="var(--mono)"
                        fontWeight="600"
                      >
                        {neuron.origIndex}
                      </text>
                    )}
                  </g>
                )
              })}

              {/* Ellipsis for truncated large layers */}
              {layer.display.hasEllipsis && (
                <text
                  x={layer.x}
                  y={svgHeight / 2}
                  textAnchor="middle"
                  fill="#666"
                  fontSize="18"
                  fontWeight="bold"
                  fontFamily="var(--mono)"
                >
                  ⋮
                </text>
              )}

              {/* Layer size inside the rectangle */}
              <text
                x={layer.x}
                y={(svgHeight - layer.h) / 2 - 20}
                textAnchor="middle"
                fill="#666"
                fontSize="13"
                fontFamily="var(--mono)"
                fontWeight="600"
              >
                {layer.size}
              </text>

              {/* Layer name label */}
              <text
                x={layer.x}
                y={svgHeight - 8}
                textAnchor="middle"
                fill="#333"
                fontSize="13"
                fontFamily="var(--sans)"
              >
                {layer.label}
              </text>

              {/* Activation function label (e.g., "relu", "softmax") */}
              {actLabel && (
                <text
                  x={layer.x}
                  y={10}
                  textAnchor="middle"
                  fill={actLabel === 'softmax' ? '#4ecdc4' : '#e94560'}
                  fontSize="12"
                  fontFamily="var(--mono)"
                  opacity={0.7}
                >
                  {actLabel}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
