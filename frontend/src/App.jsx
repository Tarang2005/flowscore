/**
 * FlowScore — App Root
 * ====================
 * Main application shell with:
 *   - Header with branding
 *   - Persona selector
 *   - Tab navigation (Borrower View / Lender View)
 *   - Footer
 */

import { useState, useEffect, useCallback } from 'react';
import { fetchPersonas, fetchBorrower } from './services/api';

import PersonaSelector from './components/PersonaSelector';
import Dashboard from './pages/Dashboard';
import LenderView from './pages/LenderView';

const TABS = [
  { id: 'borrower', label: 'Borrower View', icon: '👤' },
  { id: 'lender', label: 'Lender View', icon: '🏦' },
];

export default function App() {
  // ── State ──
  const [activeTab, setActiveTab] = useState('borrower');
  const [personas, setPersonas] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [borrower, setBorrower] = useState(null);
  const [loading, setLoading] = useState(false);
  const [personasLoading, setPersonasLoading] = useState(true);
  const [error, setError] = useState(null);

  // ── Load personas on mount ──
  useEffect(() => {
    async function loadPersonas() {
      try {
        setPersonasLoading(true);
        const data = await fetchPersonas();
        setPersonas(data);

        // Auto-select Priya
        if (data.length > 0) {
          setSelectedId(data[0].borrower_id);
        }
      } catch (err) {
        console.error('Failed to load personas:', err);
        setError('Failed to connect to FlowScore API. Is the backend running?');
      } finally {
        setPersonasLoading(false);
      }
    }

    loadPersonas();
  }, []);

  // ── Fetch borrower when selection changes ──
  const loadBorrower = useCallback(async (id) => {
    if (!id) return;

    setLoading(true);
    setError(null);

    try {
      const data = await fetchBorrower(id);
      setBorrower(data);
    } catch (err) {
      console.error(`Failed to load borrower ${id}:`, err);
      setError(err.message || 'Failed to load borrower data');
      setBorrower(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedId) {
      loadBorrower(selectedId);
    }
  }, [selectedId, loadBorrower]);

  // ── Handler ──
  function handlePersonaSelect(id) {
    setSelectedId(id);
  }

  return (
    <div className="min-h-screen flex flex-col bg-surface-900">
      {/* ════════════════ Header ════════════════ */}
      <header className="sticky top-0 z-30 border-b border-surface-700/50 bg-surface-900/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/20">
                <span className="text-white font-black text-sm">FS</span>
              </div>
              <div>
                <h1 className="text-lg font-bold text-surface-100 tracking-tight">
                  FlowScore
                </h1>
                <p className="text-[10px] text-surface-500 -mt-0.5 tracking-wider uppercase">
                  AI Credit Scoring
                </p>
              </div>
            </div>

            {/* Status badge */}
            <div className="hidden sm:flex items-center gap-2 text-xs text-surface-400">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              API Connected
            </div>
          </div>
        </div>
      </header>

      {/* ════════════════ Main Content ════════════════ */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Persona Selector */}
        <div className="mb-6 max-w-xl">
          <label className="block text-xs text-surface-400 uppercase tracking-wider font-semibold mb-2">
            Select Borrower
          </label>
          <PersonaSelector
            personas={personas}
            selectedId={selectedId}
            onSelect={handlePersonaSelect}
            loading={personasLoading}
          />
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 mb-6 border-b border-surface-700/50">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center gap-2 px-4 py-3 text-sm font-medium
                transition-all duration-200 relative
                ${activeTab === tab.id
                  ? 'text-primary-400 tab-active'
                  : 'text-surface-400 hover:text-surface-200'
                }
              `}
              id={`tab-${tab.id}`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="pb-8">
          {activeTab === 'borrower' ? (
            <Dashboard
              borrower={borrower}
              loading={loading}
              error={error}
            />
          ) : (
            <LenderView
              borrower={borrower}
              loading={loading}
              error={error}
            />
          )}
        </div>
      </main>

      {/* ════════════════ Footer ════════════════ */}
      <footer className="border-t border-surface-700/50 bg-surface-900/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-surface-500">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-surface-400">FlowScore v1.0</span>
              <span>•</span>
              <span>AI Credit Scoring for Gig Workers</span>
            </div>
            <div className="flex items-center gap-4">
              <span>XGBoost + SHAP</span>
              <span>•</span>
              <span>FastAPI + React</span>
              <span>•</span>
              <a
                href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-400 hover:text-primary-300 transition-colors"
              >
                API Docs ↗
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
