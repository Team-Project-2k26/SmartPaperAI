// Chat.jsx — standalone chat page (not used in the primary flow,
// which embeds the chatbot in Analysis.jsx)
// Kept here as a placeholder to satisfy the existing file structure.
import { useNavigate } from 'react-router-dom';
import React, { useEffect } from 'react';

export default function Chat() {
  const navigate = useNavigate();
  useEffect(() => { navigate('/'); }, [navigate]);
  return null;
}
