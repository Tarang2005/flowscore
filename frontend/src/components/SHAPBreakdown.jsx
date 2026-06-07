/**
 * SHAPBreakdown — Horizontal bar chart showing feature contributions.
 *
 * Props:
 *   positiveFactors  {Array}  [{feature, contribution, value}, ...]
 *   negativeFactors  {Array}  [{feature, contribution, value}, ...]
 */

import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, LabelList,
} from 'recharts';

function formatFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/6m/g, '(6M)')
    .replace(/3m/g, '(3M)')
    .replace(/\bpct\b/g, '%')
    .replace(/\bavg\b/g, 'Avg')
    .replace(/\bstd dev\b/g, 'Std Dev')
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;

  return (
    <div className="bg-surface-800 border border-surface-600 rounded-lg px-4 py-3 shadow-xl">
      <p className="text-sm font-semibold text-surface-100 mb-1">
        {formatFeatureName(d.feature)}
      </p>
      <p className="text-xs text-surface-400">
        Value: <span className="text-surface-200 font-mono">{d.rawValue}</span>
      </p>
      <p className="text-xs mt-1">
        Impact:{' '}
        <span className={`font-semibold ${d.contribution > 0 ? 'text-red-400' : 'text-green-400'}`}>
          {d.contribution > 0 ? '+' : ''}{d.contribution.toFixed(1)}
        </span>
      </p>
    </div>
  );
}

export default function SHAPBreakdown({ positiveFactors = [], negativeFactors = [] }) {
  // Positive factors increase default risk (bad for borrower)
  // Negative factors decrease default risk (good for borrower)
  const data = [
    ...negativeFactors.map(f => ({
      feature: f.feature,
      contribution: f.contribution,
      rawValue: f.value,
      type: 'positive', // Positive for the borrower (green)
    })),
    ...positiveFactors.map(f => ({
      feature: f.feature,
      contribution: f.contribution,
      rawValue: f.value,
      type: 'negative', // Negative for the borrower (red)
    })),
  ].sort((a, b) => b.contribution - a.contribution);

  const hasData = data.length > 0;

  return (
    <div className="glass-card p-6" id="shap-breakdown">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider">
          🔍 Score Factors (SHAP)
        </h3>
        <div className="flex items-center gap-3 text-[10px]">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-400" /> Helps Score
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-400" /> Hurts Score
          </span>
        </div>
      </div>

      {!hasData ? (
        <div className="h-48 flex items-center justify-center text-surface-500 text-sm">
          No SHAP data available — model not loaded
        </div>
      ) : (
        <div className="space-y-2">
          {data.map((item, i) => {
            const isGood = item.contribution < 0;
            const absVal = Math.abs(item.contribution);
            const maxAbs = Math.max(...data.map(d => Math.abs(d.contribution)));
            const barWidth = maxAbs > 0 ? (absVal / maxAbs) * 100 : 0;

            return (
              <div key={item.feature} className="group" style={{ animationDelay: `${i * 50}ms` }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-surface-300 font-medium truncate max-w-[60%]">
                    {formatFeatureName(item.feature)}
                  </span>
                  <span className={`text-xs font-mono font-semibold ${
                    isGood ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {item.contribution > 0 ? '+' : ''}{item.contribution.toFixed(1)}
                  </span>
                </div>
                <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ease-out ${
                      isGood
                        ? 'bg-gradient-to-r from-green-500 to-green-400'
                        : 'bg-gradient-to-r from-red-500 to-red-400'
                    }`}
                    style={{ width: `${barWidth}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
