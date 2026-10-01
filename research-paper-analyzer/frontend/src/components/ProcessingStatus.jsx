import React from 'react';

const STEPS = [
  { id: 'upload', label: 'PDF uploaded successfully', icon: '✅' },
  { id: 'extract', label: 'Extracting text from PDF', icon: '📖' },
  { id: 'sections', label: 'Detecting paper sections', icon: '🗂️' },
  { id: 'nlp', label: 'Running NLP preprocessing', icon: '⚙️' },
  { id: 'summary', label: 'Generating structured summary', icon: '📝' },
  { id: 'insight', label: 'Extracting key insights', icon: '💡' },
  { id: 'future', label: 'Generating future research directions', icon: '🔭' },
  { id: 'qa', label: 'Setting up Q&A system', icon: '💬' },
];

export default function ProcessingStatus({ currentStep = 0 }) {
  return (
    <div className="processing-container">
      <div className="spinner-ring" aria-label="Processing" role="status" />

      <div>
        <h3 style={{ marginBottom: '0.5rem', textAlign: 'center' }}>
          Analyzing Paper...
        </h3>
        <p className="text-muted" style={{ textAlign: 'center', fontSize: '0.875rem' }}>
          Open-source AI models are processing your research paper.
          <br />This may take 30–120 seconds on the first run.
        </p>
      </div>

      <div className="processing-steps" role="list">
        {STEPS.map((step, idx) => {
          const isDone = idx < currentStep;
          const isActive = idx === currentStep;
          return (
            <div
              key={step.id}
              className={`step-item ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}
              role="listitem"
              aria-current={isActive ? 'step' : undefined}
            >
              <span aria-hidden="true">
                {isDone ? '✅' : isActive ? '⏳' : '○'}
              </span>
              <span>{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
