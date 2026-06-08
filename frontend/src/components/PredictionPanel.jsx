import React from 'react';
import { HelpCircle, BarChart2 } from 'lucide-react';

export default function PredictionPanel({ result }) {
  if (!result) {
    return (
      <div className="empty-state">
        <HelpCircle size={36} strokeWidth={1.5} />
        <p>Draw a digit or upload an image, then click Predict to view model analytics.</p>
      </div>
    );
  }

  const { prediction, confidence, probabilities, preprocessed_image } = result;

  return (
    <div className="results-container">
      {/* Top highlight */}
      <div className="highlight-box">
        <div className="digit-display">{prediction}</div>
        <div className="meta-display">
          <h3>Predicted Digit: {prediction}</h3>
          <p>
            Confidence:{' '}
            <span className="confidence-val">{confidence}%</span>
          </p>
        </div>
      </div>

      {/* Probability bars */}
      <div>
        <p className="chart-section-title">
          <BarChart2 size={12} style={{ display: 'inline', marginRight: 4 }} />
          Probability Distribution
        </p>
        <div className="chart-container">
          {probabilities.map((prob, idx) => {
            const isWinner = idx === prediction;
            const percentage = (prob * 100).toFixed(1);
            return (
              <div key={idx} className={`chart-row ${isWinner ? 'winner' : ''}`}>
                <span className="chart-label">{idx}</span>
                <div className="chart-bar-bg">
                  <div
                    className={`chart-bar-fill ${isWinner ? 'high-prob' : ''}`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <span className="chart-value">{percentage}%</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* CNN input preview */}
      {preprocessed_image && (
        <div className="preprocess-section">
          <h4>CNN Input — 28×28</h4>
          <div className="preprocess-display">
            <img
              src={preprocessed_image}
              alt="CNN 28×28 input"
              className="pixel-grid-preview"
            />
            <div className="preprocess-info">
              <p>
                The backend resizes, grayscales, and normalizes the input to 28×28px
                with white strokes on black — matching the MNIST training format.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
