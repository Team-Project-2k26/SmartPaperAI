import React from 'react';

/**
 * Renders a single chat message bubble.
 *
 * Props:
 *   role - 'user' | 'ai' | 'error'
 *   content - message text
 *   sources - optional array of {section, pages, similarity}
 *   isTyping - show typing indicator instead of content
 */
export default function ChatMessage({ role, content, sources = [], isTyping = false }) {
  const isUser = role === 'user';
  const isError = role === 'error';

  return (
    <div className={`message-row ${isUser ? 'user' : ''}`} role="listitem">
      {/* Avatar */}
      <div
        className={`message-avatar ${isUser ? 'user' : 'ai'}`}
        aria-hidden="true"
      >
        {isUser ? '👤' : isError ? '⚠️' : '🤖'}
      </div>

      {/* Bubble */}
      <div className={`message-bubble ${isError ? 'error' : isUser ? 'user' : 'ai'}`}>
        {isTyping ? (
          <div className="flex items-center gap-1" aria-label="AI is thinking">
            <div className="pulse-dot" style={{ animationDelay: '0ms' }} />
            <div className="pulse-dot" style={{ animationDelay: '200ms' }} />
            <div className="pulse-dot" style={{ animationDelay: '400ms' }} />
          </div>
        ) : (
          <>
            <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {content}
            </div>

            {/* Source references */}
            {sources && sources.length > 0 && (
              <div className="message-sources" role="list" aria-label="Source passages">
                {sources.map((src, idx) => (
                  <span key={idx} className="source-chip" role="listitem">
                    📄 {src.section || 'unknown'}
                    {src.pages && src.pages.length > 0 && ` · p.${src.pages.join(', ')}`}
                  </span>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
