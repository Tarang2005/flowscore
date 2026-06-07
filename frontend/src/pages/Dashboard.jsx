/**
 * Dashboard — Main borrower-facing view.
 *
 * Layout:
 *   Left col:  ScoreGauge
 *   Right col: IncomeChart → SHAPBreakdown → CoachingTips
 *
 * Props:
 *   borrower   {Object|null}  BorrowerResponse from API
 *   loading    {boolean}
 *   error      {string|null}
 */

import ScoreGauge from '../components/ScoreGauge';
import IncomeChart from '../components/IncomeChart';
import SHAPBreakdown from '../components/SHAPBreakdown';
import CoachingTips from '../components/CoachingTips';

function LoadingSkeleton() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-pulse">
      <div className="lg:col-span-1 space-y-6">
        <div className="skeleton h-72 rounded-2xl" />
        <div className="skeleton h-28 rounded-2xl" />
      </div>
      <div className="lg:col-span-2 space-y-6">
        <div className="skeleton h-64 rounded-2xl" />
        <div className="skeleton h-52 rounded-2xl" />
        <div className="skeleton h-48 rounded-2xl" />
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
      <p className="text-xs text-surface-500 mt-4">
        Make sure the backend is running at <code className="text-primary-400">{import.meta.env.VITE_API_URL || 'localhost:8000'}</code>
      </p>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="glass-card p-12 text-center">
      <div className="text-5xl mb-4">👆</div>
      <h3 className="text-lg font-semibold text-surface-200 mb-2">Select a Persona</h3>
      <p className="text-sm text-surface-400 max-w-sm mx-auto">
        Choose one of the demo borrowers above to view their FlowScore dashboard
        with income charts, SHAP explanations, and coaching tips.
      </p>
    </div>
  );
}

export default function Dashboard({ borrower, loading, error }) {
  if (loading) return <LoadingSkeleton />;
  if (error) return <ErrorState message={error} />;
  if (!borrower) return <EmptyState />;

  const {
    flowscore,
    risk_category,
    score_history = [],
    profile = {},
    shap_explanation = {},
    coaching_tips = [],
  } = borrower;

  // Get previous score from history for delta
  const prevScore = score_history.length >= 2
    ? score_history[score_history.length - 2].score
    : null;

  // Platform data for income chart
  const platforms = profile?.income_data?.platforms || [];

  // SHAP factors
  const posFactors = shap_explanation?.top_positive_factors || [];
  const negFactors = shap_explanation?.top_negative_factors || [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* ── Left Column: Score + Stats ── */}
      <div className="lg:col-span-1 space-y-6">
        {/* Score Gauge */}
        <div className="animate-fade-in-up">
          <ScoreGauge
            score={flowscore}
            prevScore={prevScore}
            riskCategory={risk_category}
          />
        </div>

        {/* Quick Stats */}
        <div className="glass-card p-5 animate-fade-in-up-delay-1">
          <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-3">
            Profile Summary
          </h3>
          <div className="space-y-3">
            <StatRow
              label="Monthly Income"
              value={`₹${(profile?.income_data?.total_monthly_income_current || 0).toLocaleString('en-IN')}`}
            />
            <StatRow
              label="Monthly Spending"
              value={`₹${(profile?.spending_data?.avg_monthly_spending || 0).toLocaleString('en-IN')}`}
            />
            <StatRow
              label="Spending Ratio"
              value={`${((profile?.calculated_features?.spending_to_income_ratio || 0) * 100).toFixed(0)}%`}
              highlight={profile?.calculated_features?.spending_to_income_ratio > 0.7}
            />
            <StatRow
              label="Platforms"
              value={platforms.map(p => p.platform.replace(/_/g, ' ')).join(', ')}
            />
            <StatRow
              label="Late Payments"
              value={profile?.spending_data?.late_payments_count_6m || 0}
              highlight={(profile?.spending_data?.late_payments_count_6m || 0) > 0}
            />
          </div>
        </div>

        {/* Score History */}
        {score_history.length > 0 && (
          <div className="glass-card p-5 animate-fade-in-up-delay-2">
            <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-wider mb-3">
              Score History
            </h3>
            <div className="space-y-2">
              {score_history.slice(-4).map((entry, i) => (
                <div key={i} className="flex justify-between items-center text-sm">
                  <span className="text-surface-400 text-xs">{entry.date}</span>
                  <span className="font-mono font-semibold text-surface-200">{entry.score}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Right Column: Charts + Tips ── */}
      <div className="lg:col-span-2 space-y-6">
        {/* Income Chart */}
        <div className="animate-fade-in-up-delay-1">
          <IncomeChart platforms={platforms} />
        </div>

        {/* SHAP Breakdown */}
        <div className="animate-fade-in-up-delay-2">
          <SHAPBreakdown
            positiveFactors={posFactors}
            negativeFactors={negFactors}
          />
        </div>

        {/* Coaching Tips */}
        <div className="animate-fade-in-up-delay-3">
          <CoachingTips tips={coaching_tips} />
        </div>
      </div>
    </div>
  );
}

/* ── Tiny helper ── */
function StatRow({ label, value, highlight = false }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-surface-400">{label}</span>
      <span className={`text-sm font-medium ${
        highlight ? 'text-amber-400' : 'text-surface-200'
      }`}>
        {value}
      </span>
    </div>
  );
}
