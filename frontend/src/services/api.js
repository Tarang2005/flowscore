/**
 * FlowScore API Service
 * =====================
 * Centralized API client for all backend communication.
 * Uses axios with base URL from environment.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ── Response interceptor for consistent error handling ──

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.error ||
      error.message ||
      'An unexpected error occurred';

    console.error(`[FlowScore API] ${error.config?.method?.toUpperCase()} ${error.config?.url} → ${error.response?.status || 'NETWORK_ERROR'}: ${message}`);

    return Promise.reject({
      status: error.response?.status || 0,
      message,
      raw: error,
    });
  }
);

// ── API Methods ──

/**
 * Fetch the list of demo personas.
 * @returns {Promise<Array>} List of persona objects with expected scores
 */
export async function fetchPersonas() {
  const { data } = await api.get('/demo/personas');
  return data.personas || [];
}

/**
 * Fetch a borrower's full profile, score, and history.
 * @param {string} borrowerId - e.g. "priya_001"
 * @returns {Promise<Object>} BorrowerResponse
 */
export async function fetchBorrower(borrowerId) {
  const { data } = await api.get(`/borrower/${borrowerId}`);
  return data;
}

/**
 * Score a borrower profile.
 * @param {Object} borrowerProfile - Full BorrowerScoreRequest body
 * @returns {Promise<Object>} ScoreResponse
 */
export async function fetchScore(borrowerProfile) {
  const { data } = await api.post('/score', borrowerProfile);
  return data;
}

/**
 * Ingest a mock Razorpay/UPI transaction.
 * @param {string} borrowerId
 * @param {Object} transaction - Transaction payload
 * @returns {Promise<Object>} IngestResponse
 */
export async function ingestTransaction(borrowerId, transaction) {
  const { data } = await api.post('/ingest', {
    borrower_id: borrowerId,
    event: 'payment.authorized',
    transaction,
  });
  return data;
}

/**
 * Health check.
 * @returns {Promise<Object>} { status, model_loaded, version }
 */
export async function checkHealth() {
  const { data } = await api.get('/health');
  return data;
}

export default api;
