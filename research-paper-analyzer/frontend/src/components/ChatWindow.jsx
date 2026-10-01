import React, { useEffect, useRef, useState } from 'react';
import ChatMessage from './ChatMessage';
import { askQuestion } from '../services/api';

const SUGGESTIONS = [
  'What is the main objective of this paper?',
  'What dataset was used?',
  'What methodology did the authors use?',
  'What are the main findings?',
  'What are the limitations?',
  'What is the key contribution?',
  'What future work is suggested?',
  'Explain the methodology simply.',
];

/**
 * Full chat window component with message history and input.
 *
 * Props:
 *   documentId - string
 */
export default function ChatWindow({ documentId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const sendMessage = async (question) => {
    const q = (question || input).trim();
    if (!q || isLoading) return;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: q }]);
    setIsLoading(true);

    try {
      const response = await askQuestion(documentId, q);
      setMessages((prev) => [
        ...prev,
        {
          role: response.found_in_paper ? 'ai' : 'ai',
          content: response.answer,
          sources: response.sources || [],
        },
      ]);
    } catch (err) {
      const detail = err.response?.data?.detail || 'Failed to get an answer. Please try again.';
      setMessages((prev) => [...prev, { role: 'error', content: detail }]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-container" role="log" aria-label="Paper Q&A chat">
      {/* Messages area */}
      <div className="chat-messages" role="list">
        {messages.length === 0 && !isLoading ? (
          <div className="chat-empty">
            <div style={{ fontSize: '2.5rem' }}>💬</div>
            <div>
              <p style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                Ask anything about the paper
              </p>
              <p style={{ fontSize: '0.85rem' }}>
                Answers are grounded in the uploaded document
              </p>
            </div>

            <div className="chat-suggestions" role="list" aria-label="Suggested questions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className="suggestion-chip"
                  onClick={() => sendMessage(s)}
                  role="listitem"
                  aria-label={`Ask: ${s}`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, idx) => (
              <ChatMessage
                key={idx}
                role={msg.role}
                content={msg.content}
                sources={msg.sources}
              />
            ))}
            {isLoading && <ChatMessage role="ai" isTyping />}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input row */}
      <div className="chat-input-row">
        <textarea
          ref={inputRef}
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about the paper…"
          rows={1}
          disabled={isLoading}
          aria-label="Question input"
          id="chat-question-input"
        />
        <button
          id="send-question-btn"
          className="btn btn-primary btn-icon"
          onClick={() => sendMessage()}
          disabled={!input.trim() || isLoading}
          aria-label="Send question"
          title="Send (Enter)"
        >
          {isLoading ? (
            <div className="spinner-ring" style={{ width: 18, height: 18, borderWidth: 2 }} />
          ) : '➤'}
        </button>
      </div>
    </div>
  );
}
