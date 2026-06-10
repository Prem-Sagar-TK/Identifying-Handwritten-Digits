import React from 'react';
import { HelpCircle } from 'lucide-react';

export default function PredictionPanel({ result }) {
  if (!result) {
    return (
      <div className="empty-state">
        <HelpCircle size={32} strokeWidth={1.5} />
        <p>Draw a digit or upload an image, then click Predict to see results.</p>
      </div>
    );
  }

  const { prediction, preprocessed_image } = result;

  return (
    <div className="result-card results-container">
      {/* Header highlight */}
      <div className="result-card-header">
        <div className="result-digit-display">{prediction}</div>
        <div className="result-meta">
          <h3>Predicted Digit: {prediction}</h3>
          <p>The model identified this as a "{prediction}"</p>
        </div>
      </div>

      {/* CNN 28×28 preview */}
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
                The backend resizes, grayscales, and normalizes the input to 28×28 px
                with white strokes on black — matching the MNIST training format.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
