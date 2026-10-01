import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300000, // 5 min — model inference can be slow on first load
});

/**
 * Upload a PDF file.
 * @param {File} file
 * @param {function} onUploadProgress - optional progress callback
 * @returns {Promise<{document_id, filename, total_pages, is_scanned, warning}>}
 */
export async function uploadPDF(file, onUploadProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  });
  return data;
}

/**
 * Run full analysis on an uploaded document.
 * @param {string} documentId
 * @returns {Promise<AnalysisResponse>}
 */
export async function analyzeDocument(documentId) {
  const { data } = await api.post('/analyze', { document_id: documentId });
  return data;
}

/**
 * Ask a question about a paper.
 * @param {string} documentId
 * @param {string} question
 * @returns {Promise<{question, answer, found_in_paper, confidence, sources}>}
 */
export async function askQuestion(documentId, question) {
  const { data } = await api.post('/chat', {
    document_id: documentId,
    question,
  });
  return data;
}

export default api;
