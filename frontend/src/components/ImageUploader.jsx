import React, { useState, useRef } from 'react';
import { Upload, Trash2, Send } from 'lucide-react';

export default function ImageUploader({ onPredict, isLoading }) {
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (PNG or JPG).');
      return;
    }
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadend = () => setPreviewUrl(reader.result);
    reader.readAsDataURL(file);
  };

  const clearFile = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handlePredict = (e) => {
    e.stopPropagation();
    if (selectedFile) onPredict(selectedFile);
  };

  const triggerFileInput = () => fileInputRef.current.click();

  return (
    <div className="canvas-wrapper">
      <input
        ref={fileInputRef}
        type="file"
        style={{ display: 'none' }}
        accept="image/*"
        onChange={handleFileChange}
      />

      {!previewUrl ? (
        <div
          className={`uploader-box ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <div className="uploader-icon">
            <Upload size={24} />
          </div>
          <div className="upload-text">
            <h3>Drag & drop your digit image</h3>
            <p>Supports PNG, JPG · high contrast works best</p>
          </div>
          <button type="button" className="action-btn">
            Browse File
          </button>
        </div>
      ) : (
        <div className="preview-container">
          <img src={previewUrl} alt="Preview" className="preview-image" />
        </div>
      )}

      {previewUrl && (
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', justifyContent: 'flex-end', width: '100%' }}>
          <button
            onClick={clearFile}
            className="action-btn danger"
            disabled={isLoading}
          >
            <Trash2 size={13} />
            Remove
          </button>
          <button
            onClick={handlePredict}
            className="send-btn"
            title="Run prediction"
            disabled={isLoading}
          >
            <Send size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
