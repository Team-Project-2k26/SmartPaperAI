import React from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();
  const isOnAnalysis = location.pathname.startsWith('/analysis');

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <Link to="/" className="navbar-brand">
        <div className="navbar-logo-icon" aria-hidden="true">🔬</div>
        <div>
          <span className="navbar-title">SmartPaperAI</span>
          <span className="navbar-subtitle">Research Paper Analyzer</span>
        </div>
      </Link>

      <div className="flex items-center gap-2">
        {isOnAnalysis && (
          <Link to="/" className="btn btn-ghost btn-sm">
            ← Upload New
          </Link>
        )}
        <span className="navbar-badge">Open-Source AI</span>
      </div>
    </nav>
  );
}
