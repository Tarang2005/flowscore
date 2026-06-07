/**
 * Dashboard — Main borrower-facing view.
 * Refactored for the FlowScore "AI + Trust" Design System
 */

import { useState } from 'react';
import ScoreGauge from '../components/ScoreGauge';
import IncomeChart from '../components/IncomeChart';
import SHAPBreakdown from '../components/SHAPBreakdown';
import CoachingTips from '../components/CoachingTips';

function LoadingSkeleton() {
  return (
    <div className="grid grid-cols-1 xl:grid-cols-[1fr_2fr] gap-6 animate-pulse">
      <div className="space-y-6">
        <div className="skeleton h-96 rounded-2xl" />
        <div className="skeleton h-48 rounded-2xl" />
      </div>
      <div className="space-y-6">
        <div className="skeleton h-72 rounded-2xl" />
        <div className="skeleton h-48 rounded-2xl" />
        <div className="skeleton h-24 rounded-2xl" />
      </div>
    </div>
  );
}

function ErrorState({ message }) {
  return (
    <div className="glass-card p-8 text-center">
      <div className="text-4xl mb-4">⚠️</div>
      <h3 className="text-lg font-semibold text-surface-200 mb-2">Failed to Load</h3>
      <p className="text-sm text-surface-400 max-w-md mx-auto">{message}</p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="glass-card p-12 text-center border-dashed border-surface-600">
      <div className="text-5xl mb-4">👆</div>
      <h3 className="text-lg font-semibold text-surface-200 mb-2">Select a Persona</h3>
      <p className="text-sm text-surface-400 max-w-sm mx-auto">
        Choose a demo borrower above to view their FlowScore dashboard.
      </p>
    </div>
  );
}

export default function Dashboard({ borrower, loading, error }) {
  const [showJson, setShowJson] = useState(false);

  if (loading) return <LoadingSkeleton />;
  if (error) return <ErrorState message={error} />;
  if (!borrower) return <EmptyState />;

  const {
    flowscore,
    risk_category,
    profile = {},
    shap_explanation = {},
    coaching_tips = [],
  } = borrower;

  const platforms = profile?.income_data?.platforms || [];
  const posFactors = shap_explanation?.top_positive_factors || [];
  const negFactors = shap_explanation?.top_negative_factors || [];

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[400px_1fr] gap-6">
      
      {/* ── Left Column: Score + Coaching ── */}
      <div className="space-y-6">
        {/* Metric 1: Credit Tracker */}
        <div className="animate-fade-in-up">
          <ScoreGauge
            score={flowscore}
            riskCategory={risk_category}
          />
        </div>

        {/* Metric 2: Coaching */}
        <div className="animate-fade-in-up-delay-1">
          <CoachingTips tips={coaching_tips} />
        </div>
      </div>

      {/* ── Right Column: Charts, Factors & Logs ── */}
      <div className="space-y-6 min-w-0">
        
        {/* Metric 3: Income Spline */}
        <div className="animate-fade-in-up-delay-1">
          <IncomeChart platforms={platforms} />
        </div>

        {/* Metric 4: SHAP Explainability */}
        <div className="animate-fade-in-up-delay-2">
          <SHAPBreakdown
            positiveFactors={posFactors}
            negativeFactors={negFactors}
          />
        </div>

        {/* Metric 5: API Response Logging */}
        <div className="glass-card p-4 animate-fade-in-up-delay-3 flex flex-col justify-center">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-widest flex items-center gap-2">
              <span className="font-mono text-surface-500">{`{ }`}</span> API Response (SCORERESPOND)
            </h3>
            <div className="flex gap-3">
              <button 
                onClick={() => setShowJson(!showJson)}
                className="text-xs bg-[rgba(255,255,255,0.03)] hover:bg-[rgba(255,255,255,0.08)] border border-[rgba(255,255,255,0.05)] px-3 py-1.5 rounded-md transition-colors"
              >
                {showJson ? 'Hide JSON' : 'Show JSON'}
              </button>
              <button 
                onClick={() => navigator.clipboard.writeText(JSON.stringify(borrower, null, 2))}
                className="text-xs bg-[rgba(255,255,255,0.03)] hover:bg-[rgba(255,255,255,0.08)] border border-[rgba(255,255,255,0.05)] px-3 py-1.5 rounded-md transition-colors"
              >
                Copy Raw JSON
              </button>
            </div>
          </div>
          
          {showJson && (
            <div className="mt-3 bg-[#050810] border border-[rgba(255,255,255,0.03)] rounded-lg p-4 overflow-auto max-h-60 custom-scrollbar">
              <pre className="text-[11px] font-mono text-surface-400 whitespace-pre-wrap">
                {JSON.stringify(borrower, null, 2)}
              </pre>
            </div>
          )}
          {!showJson && (
            <div className="mt-1 flex items-center gap-2 text-surface-500 font-mono text-xs">
              <span className="text-green-500 font-bold">{`>`}_</span>
              <span>Command Line / Ready</span>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
