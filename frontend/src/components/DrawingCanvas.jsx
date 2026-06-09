import React, { useRef, useState, useEffect } from 'react';
import { Eraser, Brain } from 'lucide-react';

export default function DrawingCanvas({ onPredict, isLoading }) {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushWidth, setBrushWidth] = useState(16);
  const lastX = useRef(0);
  const lastY = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      // Initialize with solid black background
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();

    // For touch events
    if (e.touches && e.touches.length > 0) {
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top
      };
    }

    // For mouse events
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    const { x, y } = getCoordinates(e);
    lastX.current = x;
    lastY.current = y;
    setIsDrawing(true);

    // Draw a single dot immediately (isDrawing state update is async so we
    // can't rely on it being true inside draw() yet — draw the dot directly).
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = brushWidth;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.1, y + 0.1);   // tiny offset so lineTo registers a dot
    ctx.stroke();
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { x, y } = getCoordinates(e);

    ctx.strokeStyle = '#FFFFFF';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = brushWidth;

    ctx.beginPath();
    ctx.moveTo(lastX.current, lastY.current);
    ctx.lineTo(x, y);
    ctx.stroke();

    lastX.current = x;
    lastY.current = y;
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const handlePredict = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], 'digit.png', { type: 'image/png' });
        onPredict(file);
      }
    }, 'image/png');
  };

  return (
    <div className="canvas-wrapper">
      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
      </div>

      <div className="canvas-controls">
        <div className="brush-slider-group">
          <span>Brush Size:</span>
          <input
            type="range"
            min="8"
            max="28"
            value={brushWidth}
            onChange={(e) => setBrushWidth(parseInt(e.target.value))}
            className="brush-slider"
          />
          <span>{brushWidth}px</span>
        </div>

        <div className="btn-group">
          <button
            onClick={clearCanvas}
            className="btn btn-danger"
            title="Clear drawing board"
            disabled={isLoading}
          >
            Clear
          </button>

          <button
            onClick={handlePredict}
            className="btn btn-primary"
            disabled={isLoading}
          >
            Predict
          </button>
        </div>
      </div>
    </div>
  );
}
