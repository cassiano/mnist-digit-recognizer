/**
 * NetworkInfo component — displays summary statistics about the network architecture.
 *
 * Shows per-layer neuron/parameter counts and totals. Uses getNetworkInfo()
 * from the Network class to compute statistics. This is a read-only display
 * component that updates whenever the network changes.
 */
import { Network } from '../neural-network/Network';

interface NetworkInfoProps {
  network: Network;
}

export function NetworkInfo({ network }: NetworkInfoProps) {
  const info = network.getNetworkInfo();

  return (
    <div className="network-info">
      <h3>Network Architecture</h3>
      <div className="architecture">
        {info.layerDetails.map((layer, idx) => (
          <div key={idx} className="layer-info">
            <div className="layer-label">
              {idx === info.layerDetails.length - 1 ? 'Output' : `Hidden ${idx + 1}`}
            </div>
            <div className="layer-neurons">{layer.neurons} neurons</div>
            <div className="layer-weights">{layer.weights.toLocaleString()} parameters</div>
            {idx < info.layerDetails.length - 1 && <div className="layer-arrow">→</div>}
          </div>
        ))}
      </div>
      <div className="network-stats">
        <p>Total Neurons: {info.totalNeurons}</p>
        <p>Total Parameters: {info.totalWeights.toLocaleString()}</p>
      </div>
    </div>
  );
}
