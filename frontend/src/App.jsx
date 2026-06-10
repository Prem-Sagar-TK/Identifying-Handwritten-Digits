import React, { useState, useRef, useEffect } from 'react';
import { Edit3, Upload, Brain, BarChart2, Sparkles, Send, Trash2, RefreshCcw, AlertCircle } from 'lucide-react';
import DrawingCanvas from './components/DrawingCanvas';
import ImageUploader from './components/ImageUploader';
import PredictionPanel from './components/PredictionPanel';

// GitHub icon (lucide-react doesn't always export Github)
const GithubIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
  </svg>
);

export default function App() {
  const [activeTab, setActiveTab] = useState('draw');
  const [messages, setMessages] = useState([]); // { type: 'result'|'error', data/message: ... }
  const [isLoading, setIsLoading] = useState(false);
  const [canvasKey, setCanvasKey] = useState(0); // force remount to reset canvas
  const feedRef = useRef(null);

  // Auto-scroll to bottom of feed whenever messages change
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handlePredict = async (file) => {
    setIsLoading(true);

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
      setMessages(prev => [...prev, { type: 'result', data }]);
    } catch (err) {
      console.error('Prediction API Error:', err);
      setMessages(prev => [
        ...prev,
        { type: 'error', message: err.message || 'An unexpected error occurred. Is the backend running?' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
  };

  const handleNewChat = () => {
    setMessages([]);
    setCanvasKey(k => k + 1);
    setIsLoading(false);
  };

  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <>
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <a href="/" className="sidebar-logo">
            <div className="sidebar-logo-icon">D</div>
            <span className="sidebar-logo-text">Digit Recognizer</span>
          </a>
          <button className="sidebar-new-btn" title="New session" onClick={handleNewChat}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 5v14M5 12h14" />
            </svg>
          </button>
        </div>

        <div className="sidebar-section-label">Input Mode</div>

        <div className="sidebar-mode-list">
          <button
            id="sidebar-draw-btn"
            className={`sidebar-mode-item ${activeTab === 'draw' ? 'active' : ''}`}
            onClick={() => handleTabChange('draw')}
          >
            <Edit3 size={16} />
            Draw on Canvas
          </button>
          <button
            id="sidebar-upload-btn"
            className={`sidebar-mode-item ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => handleTabChange('upload')}
          >
            <Upload size={16} />
            Upload Image
          </button>
        </div>

        <div className="sidebar-divider" />

        <div className="sidebar-section-label">Model Info</div>
        <div className="sidebar-mode-list">
          <div className="sidebar-mode-item" style={{ cursor: 'default' }}>
            <Brain size={16} />
            CNN (MNIST)
            <span className="sidebar-info-tag">v1</span>
          </div>
          <div className="sidebar-mode-item" style={{ cursor: 'default' }}>
            <BarChart2 size={16} />
            10-class output
          </div>
        </div>

        <div className="sidebar-bottom">
          <a
            href="https://github.com/Prem-Sagar-TK/Identifying-Handwritten-Digits"
            target="_blank"
            rel="noopener noreferrer"
            className="sidebar-bottom-link"
          >
            <GithubIcon />
            View on GitHub
          </a>
          <button
            className="sidebar-bottom-link"
            style={{ background: 'none', border: 'none', cursor: 'pointer', width: '100%' }}
            onClick={handleNewChat}
          >
            <RefreshCcw size={16} />
            New Session
          </button>
        </div>
      </aside>

      {/* ── Main area ── */}
      <div className="main">
        {/* Top bar */}
        <header className="topbar">
          <div className="topbar-title">
            <Sparkles size={16} style={{ color: 'var(--accent)' }} />
            Handwritten Digit Recognition
            <span className="topbar-title-badge">CNN</span>
          </div>
          <div className="topbar-actions">
            <button className="topbar-btn" onClick={handleNewChat}>
              <RefreshCcw size={13} />
              New Session
            </button>
          </div>
        </header>

        {/* ── Side-by-side layout ── */}
        <div className="split-layout">

          {/* LEFT: Canvas / Uploader */}
          <div className="split-input-panel">
            <div className="input-box">
              {/* Tabs */}
              <div className="input-tabs">
                <button
                  id="tab-draw"
                  className={`input-tab ${activeTab === 'draw' ? 'active' : ''}`}
                  onClick={() => handleTabChange('draw')}
                >
                  <Edit3 size={13} />
                  Draw
                </button>
                <button
                  id="tab-upload"
                  className={`input-tab ${activeTab === 'upload' ? 'active' : ''}`}
                  onClick={() => handleTabChange('upload')}
                >
                  <Upload size={13} />
                  Upload
                </button>
              </div>

              {/* Canvas or Uploader */}
              <div className="input-content">
                {activeTab === 'draw' ? (
                  <DrawingCanvas
                    key={canvasKey}
                    onPredict={handlePredict}
                    isLoading={isLoading}
                  />
                ) : (
                  <ImageUploader
                    onPredict={handlePredict}
                    isLoading={isLoading}
                  />
                )}
              </div>
            </div>
            <p className="input-hint">
              Digits only (0–9) · Runs locally · No data is stored
            </p>
          </div>

          {/* Divider */}
          <div className="split-divider" />

          {/* RIGHT: Results / Outcome */}
          <div className="split-result-panel" ref={feedRef}>
            {messages.length === 0 && !isLoading ? (
              <div className="empty-state">
                <Brain size={40} style={{ opacity: 0.25, marginBottom: '8px' }} />
                <p>Results will appear here after you predict a digit.</p>
              </div>
            ) : (
              <div className="result-feed">
                {messages.map((msg, idx) => (
                  <div key={idx} className="message message-ai">
                    <div className="message-header">
                      <div className="message-avatar ai">
                        <Brain size={15} />
                      </div>
                      <span className="message-role">Model Result</span>
                      <span className="message-time">{now}</span>
                    </div>
                    <div className="message-body">
                      {msg.type === 'result' ? (
                        <PredictionPanel result={msg.data} />
                      ) : (
                        <div className="error-banner">
                          <AlertCircle size={16} style={{ flexShrink: 0 }} />
                          <div><strong>Error:</strong> {msg.message}</div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {/* Loading indicator */}
                {isLoading && (
                  <div className="message message-ai">
                    <div className="message-header">
                      <div className="message-avatar ai">
                        <Brain size={15} />
                      </div>
                      <span className="message-role">Model Result</span>
                    </div>
                    <div className="loading-message">
                      <div className="loading-dots">
                        <span /><span /><span />
                      </div>
                      Running CNN forward pass…
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

        </div>
      </div>
    </>
  );
}
