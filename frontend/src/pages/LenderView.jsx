/**
 * LenderView — Technical/API view for lenders showing raw ScoreResponse.
 *
 * Props:
 *   borrower   {Object|null}  BorrowerResponse from API
 *   loading    {boolean}
 *   error      {string|null}
 */

import { useState } from 'react';

function getScoreColor(score) {
  if (score >= 700) return 'text-green-400';
  if (score >= 600) return 'text-yellow-400';
  if (score >= 500) return 'text-orange-400';
  return 'text-red-400';
}

function getRiskBadge(category) {
  const styles = {
    low: 'bg-green-500/10 text-green-400 border-green-500/20',
    medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    high: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
    very_high: 'bg-red-500/10 text-red-400 border-red-500/20',
  };
  return styles[category] || styles.medium;
}

function formatFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export default function LenderView({ borrower, loading, error }) {
  const [copied, setCopied] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  if (loading) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="skeleton h-32 rounded-2xl" />
        <div className="skeleton h-64 rounded-2xl" />
        <div className="skeleton h-48 rounded-2xl" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card p-8 text-center">
        <div className="text-4xl mb-4">⚠️</div>
        <p className="text-sm text-surface-400">{error}</p>
      </div>
    );
  }

  if (!borrower) {
    return (
      <div className="glass-card p-12 text-center">
        <div className="text-5xl mb-4">🏦</div>
        <h3 className="text-lg font-semibold text-surface-200 mb-2">Lender View</h3>
        <p className="text-sm text-surface-400">
          Select a borrower to view the API response and credit assessment.
        </p>
      </div>
    );
  }

  const {
    borrower_id,
    flowscore,
    default_probability,
    risk_category,
    shap_explanation = {},
    coaching_tips = [],
    profile = {},
    score_history = [],
  } = borrower;

  const posFactors = shap_explanation?.top_positive_factors || [];
  const negFactors = shap_explanation?.top_negative_factors || [];

  // Build a clean ScoreResponse-like object for JSON display
  const apiResponse = {
    borrower_id,
    flowscore,
    default_probability,
    risk_category,
    shap_explanation,
    coaching_tips,
    score_history,
    model_metadata: {
      model_version: 'v1.0',
      prediction_timestamp: new Date().toISOString(),
      feature_count: 23,
    },
  };

  const jsonString = JSON.stringify(apiResponse, null, 2);

  function handleCopy() {
    navigator.clipboard.writeText(jsonString).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="space-y-6">
      {/* ── Summary Cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 animate-fade-in-up">
        <SummaryCard
          label="Borrower"
          value={profile?.name || borrower_id}
          sub={borrower_id}
          icon="👤"
        />
        <SummaryCard
          label="FlowScore"
          value={flowscore}
          icon="📊"
          valueClass={getScoreColor(flowscore)}
        />
        <SummaryCard
          label="Default Probability"
          value={`${((default_probability || 0) * 100).toFixed(1)}%`}
          icon="📉"
          valueClass={default_probability > 0.3 ? 'text-red-400' : 'text-green-400'}
        />
        <SummaryCard
          label="Risk Category"
          value={risk_category?.replace('_', ' ').toUpperCase()}
          icon="⚡"
          badge={getRiskBadge(risk_category)}
        />
      </div>

      {/* ── Loan Decision ── */}
      <div className="glass-card p-6 animate-fade-in-up-delay-1" id="loan-decision">
        <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider mb-4">
          🏦 Loan Decision
        </h3>
        <div className={`
          p-4 rounded-xl border
          ${flowscore >= 600
            ? 'bg-green-500/5 border-green-500/20'
            : 'bg-red-500/5 border-red-500/20'
          }
        `}>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl">{flowscore >= 600 ? '✅' : '❌'}</span>
            <span className={`text-lg font-bold ${flowscore >= 600 ? 'text-green-400' : 'text-red-400'}`}>
              {flowscore >= 700 ? 'APPROVED — Pre-qualified' :
               flowscore >= 600 ? 'CONDITIONALLY APPROVED' :
               'DECLINED — High Risk'}
            </span>
          </div>
          <p className="text-sm text-surface-400 ml-11">
            {flowscore >= 700
              ? `Score ${flowscore} exceeds minimum threshold. Borrower qualifies for standard rates.`
              : flowscore >= 600
              ? `Score ${flowscore} meets minimum criteria. Additional verification recommended.`
              : `Score ${flowscore} below threshold. Suggest financial coaching and reapplication in 3 months.`
            }
          </p>
        </div>
      </div>

      {/* ── SHAP Factors Table ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in-up-delay-2">
        {/* Positive (risk increasing) */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold text-red-400 uppercase tracking-wider mb-4">
            ⚠️ Risk Factors (Increases Default Risk)
          </h3>
          {posFactors.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-surface-500 uppercase">
                  <th className="text-left pb-2">Feature</th>
                  <th className="text-right pb-2">Value</th>
                  <th className="text-right pb-2">Impact</th>
                </tr>
              </thead>
              <tbody>
                {posFactors.map((f, i) => (
                  <tr key={i} className="border-t border-surface-700">
                    <td className="py-2 text-surface-300">{formatFeatureName(f.feature)}</td>
                    <td className="py-2 text-right font-mono text-surface-400">{f.value}</td>
                    <td className="py-2 text-right font-mono font-semibold text-red-400">+{f.contribution}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-surface-500 text-sm">No risk factors — model not loaded</p>
          )}
        </div>

        {/* Negative (risk decreasing) */}
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold text-green-400 uppercase tracking-wider mb-4">
            ✅ Strengths (Decreases Default Risk)
          </h3>
          {negFactors.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-surface-500 uppercase">
                  <th className="text-left pb-2">Feature</th>
                  <th className="text-right pb-2">Value</th>
                  <th className="text-right pb-2">Impact</th>
                </tr>
              </thead>
              <tbody>
                {negFactors.map((f, i) => (
                  <tr key={i} className="border-t border-surface-700">
                    <td className="py-2 text-surface-300">{formatFeatureName(f.feature)}</td>
                    <td className="py-2 text-right font-mono text-surface-400">{f.value}</td>
                    <td className="py-2 text-right font-mono font-semibold text-green-400">{f.contribution}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-surface-500 text-sm">No strength factors — model not loaded</p>
          )}
        </div>
      </div>

      {/* ── Coaching Tips ── */}
      <div className="glass-card p-6 animate-fade-in-up-delay-3" id="lender-tips">
        <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider mb-4">
          📋 Coaching Recommendations
        </h3>
        {coaching_tips.length > 0 ? (
          <ol className="space-y-2">
            {coaching_tips.map((tip, i) => (
              <li key={i} className="flex gap-3 text-sm">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-500/10 text-primary-400 flex items-center justify-center text-xs font-bold">
                  {i + 1}
                </span>
                <span className="text-surface-300">{tip}</span>
              </li>
            ))}
          </ol>
        ) : (
          <p className="text-surface-500 text-sm">No coaching tips available</p>
        )}
      </div>

      {/* ── Raw JSON ── */}
      <div className="glass-card p-6" id="raw-json">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider">
            {'{ }'} API Response (ScoreResponse)
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowRaw(!showRaw)}
              className="text-xs px-3 py-1.5 rounded-lg border border-surface-600 text-surface-400 hover:text-surface-200 hover:border-surface-500 transition-colors"
            >
              {showRaw ? 'Hide JSON' : 'Show JSON'}
            </button>
            <button
              onClick={handleCopy}
              className={`
                text-xs px-3 py-1.5 rounded-lg border transition-all duration-200
                ${copied
                  ? 'bg-green-500/10 border-green-500/30 text-green-400'
                  : 'border-surface-600 text-surface-400 hover:text-surface-200 hover:border-surface-500'
                }
              `}
            >
              {copied ? '✓ Copied!' : '📋 Copy JSON'}
            </button>
          </div>
        </div>

        {showRaw && (
          <pre className="bg-surface-900 border border-surface-700 rounded-xl p-4 text-xs font-mono text-surface-300 overflow-x-auto max-h-96">
            {jsonString}
          </pre>
        )}
      </div>
    </div>
  );
}

/* ── Summary Card Helper ── */
function SummaryCard({ label, value, sub, icon, valueClass = 'text-surface-100', badge }) {
  return (
    <div className="glass-card p-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">{icon}</span>
        <span className="text-xs text-surface-400 uppercase tracking-wider">{label}</span>
      </div>
      <div className={`text-xl font-bold ${badge || valueClass}`}>
        {badge ? (
          <span className={`inline-block px-3 py-1 rounded-lg border text-sm ${badge}`}>
            {value}
          </span>
        ) : (
          value
        )}
      </div>
      {sub && <p className="text-xs text-surface-500 mt-1 font-mono">{sub}</p>}
    </div>
  );
}
