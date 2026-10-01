import React, { useEffect, useRef, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import SummaryCard from '../components/SummaryCard';
import ChatWindow from '../components/ChatWindow';
import ProcessingStatus from '../components/ProcessingStatus';
import { analyzeDocument } from '../services/api';

export default function Analysis() {
  const { documentId } = useParams();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [step, setStep] = useState(1);
  const [error, setError] = useState(null);
  const stepRef = useRef(null);

  useEffect(() => {
    if (!documentId) { navigate('/'); return; }

    // Advance processing steps visually while analysis runs
    stepRef.current = setInterval(() => {
      setStep((s) => Math.min(s + 1, 7));
    }, 2500);

    analyzeDocument(documentId)
      .then((data) => {
        clearInterval(stepRef.current);
        setStep(8);
        setAnalysis(data);
      })
      .catch((err) => {
        clearInterval(stepRef.current);
        const msg = err.response?.data?.detail || 'Analysis failed. Please try again.';
        setError(msg);
      })
      .finally(() => setLoading(false));

    return () => clearInterval(stepRef.current);
  }, [documentId, navigate]);

  if (loading) {
    return (
      <main className="page-container">
        <ProcessingStatus currentStep={step} />
      </main>
    );
  }

  if (error) {
    return (
      <main className="page-container" style={{ textAlign: 'center', paddingTop: '4rem' }}>
        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⚠️</div>
        <h2>Analysis Failed</h2>
        <p style={{ color: 'var(--text-muted)', margin: '1rem 0 2rem' }}>{error}</p>
        <button className="btn btn-primary" onClick={() => navigate('/')}>
          ← Try Another PDF
        </button>
      </main>
    );
  }

  const {
    title, authors, year, domain, summary,
    objective, methodology, dataset, findings, main_points,
    contribution, limitations, conclusion, key_insight,
    simple_explanation, why_it_matters, keywords,
    future_research, processing_time_seconds,
  } = analysis;

  return (
    <main className="page-container" id="analysis-results">
      {/* Paper header */}
      <header style={{ marginBottom: '2rem' }}>
        <div className="paper-info-strip">
          {domain && <div className="paper-info-item">🏷️ <span>{domain}</span></div>}
          {year && <div className="paper-info-item">📅 <span>{year}</span></div>}
          {authors.length > 0 && (
            <div className="paper-info-item">
              👥 <span>{authors.slice(0, 3).join(', ')}{authors.length > 3 ? ' et al.' : ''}</span>
            </div>
          )}
          {processing_time_seconds && (
            <div className="paper-info-item">⏱️ <span>Analyzed in {processing_time_seconds}s</span></div>
          )}
        </div>

        <h1 style={{ fontSize: 'clamp(1.3rem, 3vw, 2rem)', marginBottom: '0.25rem' }}>
          {title || 'Research Paper Analysis'}
        </h1>
      </header>

      <div className="results-grid">

        {/* Summary */}
        {summary && (
          <SummaryCard icon="📋" title="Overall Summary" subtitle="AI-generated concise overview" fullWidth>
            <p className="summary-text">{summary}</p>
          </SummaryCard>
        )}

        {/* Objective */}
        {objective && (
          <SummaryCard icon="🎯" title="Research Objective">
            <p className="summary-text">{objective}</p>
          </SummaryCard>
        )}

        {/* Methodology */}
        {methodology && (
          <SummaryCard icon="⚙️" title="Methodology">
            <p className="summary-text">{methodology}</p>
          </SummaryCard>
        )}

        {/* Dataset */}
        {dataset && (
          <SummaryCard icon="🗃️" title="Dataset / Data Used">
            <p className="summary-text">{dataset}</p>
          </SummaryCard>
        )}

        {/* Main points */}
        {main_points && main_points.length > 0 && (
          <SummaryCard icon="📌" title="Main Points" subtitle="Top-ranked sentences across the paper">
            <ul className="points-list" role="list">
              {main_points.map((point, i) => (
                <li key={i}>{point}</li>
              ))}
            </ul>
          </SummaryCard>
        )}

        {/* Findings */}
        {findings && findings.length > 0 && (
          <SummaryCard icon="📊" title="Key Findings & Results">
            <ul className="findings-list" role="list">
              {findings.map((f, i) => (
                <li key={i}>{f}</li>
              ))}
            </ul>
          </SummaryCard>
        )}

        {/* Contribution */}
        {contribution && (
          <SummaryCard icon="🏆" title="Main Contribution">
            <p className="summary-text">{contribution}</p>
          </SummaryCard>
        )}

        {/* Conclusion */}
        {conclusion && (
          <SummaryCard icon="✅" title="Conclusion">
            <p className="summary-text">{conclusion}</p>
          </SummaryCard>
        )}

        {/* Key insight */}
        <SummaryCard icon="💡" title="Key Insight" subtitle="The most important contribution" fullWidth>
          {key_insight && (
            <div className="insight-block">
              <div className="insight-label">Key Insight</div>
              <div className="insight-text">{key_insight}</div>
            </div>
          )}
          {simple_explanation && (
            <div className="insight-block">
              <div className="insight-label">In Simple Language</div>
              <div className="insight-text">{simple_explanation}</div>
            </div>
          )}
          {why_it_matters && (
            <div className="insight-block" style={{ borderLeftColor: 'var(--accent-success)' }}>
              <div className="insight-label" style={{ color: 'var(--accent-success)' }}>Why It Matters</div>
              <div className="insight-text">{why_it_matters}</div>
            </div>
          )}
        </SummaryCard>

        {/* Limitations */}
        {limitations && limitations.length > 0 && (
          <SummaryCard icon="⚠️" title="Limitations">
            <ul className="findings-list" role="list">
              {limitations.map((lim, i) => (
                <li key={i} style={{ color: 'var(--text-secondary)' }}>{lim}</li>
              ))}
            </ul>
          </SummaryCard>
        )}

        {/* Keywords */}
        {keywords && keywords.length > 0 && (
          <SummaryCard icon="🏷️" title="Keywords & Topics">
            <div className="keywords-wrap" role="list" aria-label="Keywords">
              {keywords.map((kw) => (
                <span key={kw} className="keyword-tag" role="listitem">{kw}</span>
              ))}
            </div>
          </SummaryCard>
        )}

        {/* Future research */}
        {future_research && future_research.length > 0 && (
          <SummaryCard
            icon="🔭"
            title="Future Research Directions"
            subtitle="Derived from paper evidence"
            fullWidth
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {future_research.map((dir, i) => (
                <div key={i} className="future-research-item">
                  <div className="future-topic">
                    <span
                      style={{
                        background: 'var(--gradient-primary)',
                        color: 'white',
                        borderRadius: '50%',
                        width: 24,
                        height: 24,
                        display: 'inline-flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.75rem',
                        flexShrink: 0,
                      }}
                    >
                      {i + 1}
                    </span>
                    {dir.topic}
                    {dir.is_inferred && (
                      <span className="badge badge-warning" style={{ marginLeft: 'auto' }}>
                        Inferred
                      </span>
                    )}
                  </div>
                  <div className="future-question">❓ {dir.question}</div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                    {dir.reason}
                  </div>
                  {dir.evidence && (
                    <div className="future-evidence">
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: 4 }}>
                        Evidence from paper:
                      </span>
                      {dir.evidence}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </SummaryCard>
        )}

        {/* Chat with paper */}
        <SummaryCard
          icon="💬"
          title="Chat with the Paper"
          subtitle="Ask questions — answers are grounded in the document"
          fullWidth
        >
          <ChatWindow documentId={documentId} />
        </SummaryCard>
      </div>
    </main>
  );
}
