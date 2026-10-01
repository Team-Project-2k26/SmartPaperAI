import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PdfUploader from '../components/PdfUploader';
import { uploadPDF, analyzeDocument } from '../services/api';

const FEATURES = [
  { icon: '📋', label: 'Structured Summary' },
  { icon: '🎯', label: 'Research Objective' },
  { icon: '⚙️', label: 'Methodology Analysis' },
  { icon: '📊', label: 'Key Findings' },
  { icon: '💡', label: 'Core Insight' },
  { icon: '🔭', label: 'Future Directions' },
  { icon: '🏷️', label: 'Keywords & Topics' },
  { icon: '💬', label: 'Paper Q&A Chatbot' },
];

export default function Home() {
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);

  const handleFileSelected = async (file) => {
    setError(null);
    setUploading(true);
    setUploadProgress(0);

    try {
      // Step 1: Upload the PDF
      const uploadResult = await uploadPDF(file, (progressEvent) => {
        if (progressEvent.total) {
          const pct = Math.round((progressEvent.loaded / progressEvent.total) * 100);
          setUploadProgress(pct);
        }
      });

      const documentId = uploadResult.document_id;

      // Step 2: Run analysis (this is the heavy step)
      await analyzeDocument(documentId);

      // Step 3: Navigate to analysis page
      navigate(`/analysis/${documentId}`);
    } catch (err) {
      const msg =
        err.response?.data?.detail ||
        err.message ||
        'An unexpected error occurred. Please try again.';
      setError(msg);
      setUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <main className="page-container">
      {/* Hero */}
      <section className="page-hero" aria-labelledby="hero-title">
        <div className="hero-badge">
          <span>🔬</span>
          <span>Open-Source AI · No API Keys Required</span>
        </div>

        <h1 className="hero-title" id="hero-title">
          Understand Research Papers
          <br />
          <span style={{ color: 'var(--text-accent)' }}>Instantly</span>
        </h1>

        <p className="hero-subtitle">
          Upload any research paper PDF and get a structured summary, key insights,
          methodology breakdown, future research directions, and an AI chatbot — all
          powered by open-source NLP models running locally.
        </p>
      </section>

      {/* Upload card */}
      <section
        style={{ maxWidth: 680, margin: '0 auto 3rem' }}
        aria-label="PDF upload section"
      >
        <div className="card">
          <div className="card-header">
            <div className="card-icon">📤</div>
            <div>
              <div className="card-title">Upload Research Paper</div>
              <div className="card-subtitle">PDF · Max 50 MB</div>
            </div>
          </div>

          <PdfUploader
            onFileSelected={handleFileSelected}
            uploading={uploading}
            uploadProgress={uploadProgress}
            error={error}
          />
        </div>
      </section>

      {/* Features strip */}
      <section aria-label="Features">
        <div className="features-strip">
          {FEATURES.map((f) => (
            <div key={f.label} className="feature-pill">
              <span aria-hidden="true">{f.icon}</span>
              <span>{f.label}</span>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section style={{ maxWidth: 900, margin: '3rem auto 0', textAlign: 'center' }}>
        <h2 style={{ marginBottom: '2rem', color: 'var(--text-primary)' }}>How It Works</h2>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem',
          }}
        >
          {[
            { step: '1', icon: '📤', title: 'Upload PDF', desc: 'Drop your research paper' },
            { step: '2', icon: '⚙️', title: 'AI Processing', desc: 'NLP pipeline analyzes the paper' },
            { step: '3', icon: '📋', title: 'Get Insights', desc: 'Structured analysis with all sections' },
            { step: '4', icon: '💬', title: 'Ask Questions', desc: 'Chat with the paper grounded in text' },
          ].map((item) => (
            <div key={item.step} className="card" style={{ textAlign: 'center', padding: '1.5rem 1rem' }}>
              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>{item.icon}</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-accent)', fontWeight: 600, marginBottom: '0.25rem' }}>
                STEP {item.step}
              </div>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                {item.title}
              </div>
              <div className="text-muted" style={{ fontSize: '0.85rem' }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
