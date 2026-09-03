/**
 * PredictionResult component — displays the network's recognition result.
 *
 * Shows the predicted digit, confidence percentage, and a bar chart of
 * probabilities for all 10 digits. The winning digit's bar is highlighted.
 */
import type { Prediction } from '../neural-network/types';

interface PredictionResultProps {
  prediction: Prediction | null;
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  if (!prediction) return null;

  return (
    <div className="prediction-result">
      <h3>Recognition Result</h3>
      <div className="predicted-digit">
        <span className="digit">{prediction.digit}</span>
        <span className="confidence">{(prediction.confidence * 100).toFixed(2)}% confidence</span>
      </div>
      <div className="probabilities">
        <h4>Digit Probabilities</h4>
        {prediction.probabilities.map((prob, idx) => (
          <div key={idx} className="prob-bar-container">
            <span className="digit-label">{idx}</span>
            <div className="prob-bar-bg">
              <div
                className={`prob-bar ${idx === prediction.digit ? 'highlighted' : ''}`}
                style={{ width: `${prob * 100}%` }}
              />
            </div>
            <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
