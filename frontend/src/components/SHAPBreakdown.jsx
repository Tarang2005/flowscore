/**
 * SHAPBreakdown — Feature contributions.
 * Refactored for "AI + Trust" Design System
 */

function formatFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/6m/g, '')
    .replace(/3m/g, '')
    .replace(/\bpct\b/g, '%')
    .replace(/\bavg\b/g, 'Avg')
    .replace(/\bstd dev\b/g, 'Std Dev')
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
    .trim();
}

function formatValue(value, featureName) {
  if (typeof value === 'number') {
    if (featureName.includes('days') || featureName.includes('Tenure')) return `${Math.round(value)} days`;
    if (Number.isInteger(value)) return value;
    return value.toFixed(2);
  }
  return value;
}

export default function SHAPBreakdown({ positiveFactors = [], negativeFactors = [] }) {
  // Positive factors increase default risk (bad for borrower -> Negative signal)
  // Negative factors decrease default risk (good for borrower -> Positive signal)
  const data = [
    ...negativeFactors.map(f => ({
      feature: f.feature,
      rawValue: f.value,
      signal: 'Positive', 
      color: 'var(--color-signal-pos)',
    })),
    ...positiveFactors.map(f => ({
      feature: f.feature,
      rawValue: f.value,
      signal: 'Negative', 
      color: 'var(--color-signal-neg)',
    })),
  ].slice(0, 5); // Only show top 5 to keep it clean

  const hasData = data.length > 0;

  return (
    <div className="glass-card p-6" id="shap-breakdown">
      <div className="mb-4">
        <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-widest">
          Score Factors (SHAP Explainability Engine)
        </h3>
      </div>

      {!hasData ? (
        <div className="h-24 flex items-center justify-center text-surface-500 text-sm">
          No SHAP data available
        </div>
      ) : (
        <div className="space-y-0">
          {data.map((item, i) => (
            <div 
              key={i} 
              className="flex items-center justify-between py-3 border-b border-[rgba(255,255,255,0.03)] last:border-0"
            >
              <span className="text-[13px] text-surface-300">
                {formatFeatureName(item.feature)} <span className="text-surface-500">({formatValue(item.rawValue, item.feature)})</span>
              </span>
              <div className="flex items-center gap-3">
                <span className="text-[13px] text-surface-400">
                  {item.signal}
                </span>
                <span 
                  className="w-2 h-2 rounded-full" 
                  style={{ 
                    backgroundColor: item.color,
                    boxShadow: `0 0 8px ${item.color}80` 
                  }} 
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
