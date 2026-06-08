import React, { useState } from 'react';
import { Edit3, Upload, AlertCircle, Brain } from 'lucide-react';
import DrawingCanvas from './components/DrawingCanvas';
import ImageUploader from './components/ImageUploader';
import PredictionPanel from './components/PredictionPanel';

// Checkmark SVG
const Check = () => (
  <svg viewBox="0 0 20 20" fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="4 10 8 14 16 6" />
  </svg>
);

export default function App() {
  const [activeTab, setActiveTab] = useState('draw');
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (file) => {
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setPredictionResult(data);
    } catch (err) {
      console.error('Prediction API Error:', err);
      setError(err.message || 'An unexpected error occurred. Is the backend running?');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setPredictionResult(null);
    setError(null);
  };

  return (
    <>
      {/* ── Navigation ── */}
      <nav className="nav">
        <a href="/" className="nav-logo">
          Digit Recognizer
        </a>

        <ul className="nav-links">
          <li><a href="https://github.com/Prem-Sagar-TK/Identifying-Handwritten-Digits" target="_blank" rel="noopener noreferrer">GitHub</a></li>
        </ul>

        {/*<div className="nav-cta">
          <button className="btn-ghost">Log In</button>
          <button className="btn-black" onClick={() => document.querySelector('.hero-right')?.scrollIntoView({ behavior: 'smooth' })}>
            Try Now
          </button>
        </div>*/}
      </nav>

      {/* ── Hero: Split Layout ── */}
      <main className="hero">

        {/* LEFT — headline + features + CTA */}
        <div className="hero-left">


          <h1 className="hero-title">
            Identify handwritten digits instantly
          </h1>

          <ul className="feature-list">
            <li className="feature-item">
              <div className="feature-text">
                <strong>Draw directly on canvas</strong>
                <span>Sketch any digit 0 - 9 with your mouse or finger and get an instant prediction.</span>
              </div>
            </li>
            <li className="feature-item">

              <div className="feature-text">
                <strong>Upload any image</strong>
                <span>Drag & drop a PNG or JPG — the model preprocesses it automatically.</span>
              </div>
            </li>
          </ul>

          <div className="hero-cta">
            <div className="feature-text">
              No sign-up required.<br />Runs locally on your machine.
            </div>
          </div>
        </div>

        {/* RIGHT — interactive panel */}
        <div className="hero-right">
          {/* Tab bar */}
          <div className="panel-tabs">
            <button
              id="tab-draw"
              className={`panel-tab ${activeTab === 'draw' ? 'active' : ''}`}
              onClick={() => handleTabChange('draw')}
            >
              <Edit3 size={14} />
              Interactive Canvas
            </button>
            <button
              id="tab-upload"
              className={`panel-tab ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => handleTabChange('upload')}
            >
              <Upload size={14} />
              Upload Image
            </button>
          </div>

          {/* Dual column: input | output */}
          <div className="panel-columns">
            {/* Input side */}
            <div className="panel-input">
              {error && (
                <div className="error-banner">
                  <AlertCircle size={16} style={{ flexShrink: 0 }} />
                  <div><strong>Error:</strong> {error}</div>
                </div>
              )}

              {activeTab === 'draw' ? (
                <DrawingCanvas onPredict={handlePredict} isLoading={isLoading} />
              ) : (
                <ImageUploader onPredict={handlePredict} isLoading={isLoading} />
              )}
            </div>

            {/* Output side */}
            <div className="panel-output">
              <p className="panel-title">
                <Brain size={13} />
                Model Diagnostics
              </p>

              {isLoading ? (
                <div className="loading-overlay">
                  <div className="spinner" />
                  <p>Running CNN forward pass…</p>
                </div>
              ) : (
                <PredictionPanel result={predictionResult} />
              )}
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
