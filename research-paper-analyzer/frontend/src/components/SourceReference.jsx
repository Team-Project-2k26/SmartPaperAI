import React from 'react';

/**
 * Displays a source passage reference chip.
 *
 * Props:
 *   section - section name string
 *   pages - array of page numbers
 *   similarity - float 0-1
 */
export default function SourceReference({ section, pages = [], similarity }) {
  return (
    <span className="source-reference" title={`Similarity: ${(similarity * 100).toFixed(0)}%`}>
      📄 {section || 'unknown'}
      {pages && pages.length > 0 && (
        <span style={{ color: 'var(--text-muted)' }}>· p.{pages.join(', ')}</span>
      )}
    </span>
  );
}
