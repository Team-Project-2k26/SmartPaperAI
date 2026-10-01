import React, { useCallback, useRef, useState } from 'react';

const MAX_SIZE_MB = 50;

export default function PdfUploader({ onFileSelected, uploading, uploadProgress, error }) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (file.type !== 'application/pdf') {
      alert('Please upload a PDF file.');
      return;
    }
    const sizeMB = file.size / (1024 * 1024);
    if (sizeMB > MAX_SIZE_MB) {
      alert(`File size ${sizeMB.toFixed(1)} MB exceeds limit of ${MAX_SIZE_MB} MB.`);
      return;
    }
    setSelectedFile(file);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => setDragOver(false);
  const handleInputChange = (e) => handleFile(e.target.files[0]);

  const handleAnalyze = () => {
    if (selectedFile && onFileSelected) onFileSelected(selectedFile);
  };

  const handleRemove = () => {
    setSelectedFile(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div>
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !selectedFile && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload PDF file by clicking or dragging"
        onKeyDown={(e) => e.key === 'Enter' && !selectedFile && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="sr-only"
          onChange={handleInputChange}
          id="pdf-file-input"
          aria-label="Select PDF file"
        />

        {!selectedFile ? (
          <>
            <span className="upload-icon" aria-hidden="true">📄</span>
            <div className="upload-title">Drop your research paper here</div>
            <div className="upload-subtitle">
              or <span className="text-accent" style={{ cursor: 'pointer' }}>click to browse</span>
            </div>
            <div className="upload-subtitle mt-1" style={{ marginTop: '0.5rem' }}>
              Supports PDF up to {MAX_SIZE_MB} MB
            </div>
          </>
        ) : (
          <div className="upload-file-info" onClick={(e) => e.stopPropagation()}>
            <span style={{ fontSize: '1.5rem' }}>📑</span>
            <div style={{ flex: 1, textAlign: 'left' }}>
              <div style={{ fontWeight: 600, color: 'var(--accent-success)', fontSize: '0.9rem' }}>
                {selectedFile.name}
              </div>
              <div className="text-muted" style={{ fontSize: '0.8rem' }}>
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
              </div>
            </div>
            {!uploading && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={handleRemove}
                aria-label="Remove selected file"
              >
                ✕
              </button>
            )}
          </div>
        )}

        {uploading && (
          <div className="upload-progress-bar">
            <div
              className="upload-progress-fill"
              style={{ width: `${uploadProgress}%` }}
              role="progressbar"
              aria-valuenow={uploadProgress}
              aria-valuemin={0}
              aria-valuemax={100}
            />
          </div>
        )}
      </div>

      {error && (
        <div className="alert alert-error mt-2" role="alert">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {selectedFile && !uploading && (
        <button
          id="analyze-btn"
          className="btn btn-primary btn-lg w-full mt-2"
          onClick={handleAnalyze}
          style={{ marginTop: '1.25rem', justifyContent: 'center' }}
        >
          🚀 Analyze Paper
        </button>
      )}

      {uploading && (
        <div className="processing-container" style={{ padding: '1.5rem' }}>
          <div className="flex items-center gap-2">
            <div className="spinner-ring" style={{ width: 32, height: 32 }} />
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
              Uploading and processing… {uploadProgress > 0 ? `${uploadProgress}%` : ''}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
