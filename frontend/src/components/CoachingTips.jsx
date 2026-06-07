/**
 * CoachingTips — Actionable coaching cards for borrowers.
 *
 * Props:
 *   tips {Array<string>} List of coaching tip strings
 */

const TIP_ICONS = ['💡', '📊', '🎯', '🚀', '⭐'];

const TIP_STYLES = [
  {
    border: 'border-blue-500/20',
    bg: 'bg-blue-500/5',
    iconBg: 'bg-blue-500/10',
    accent: 'text-blue-400',
  },
  {
    border: 'border-emerald-500/20',
    bg: 'bg-emerald-500/5',
    iconBg: 'bg-emerald-500/10',
    accent: 'text-emerald-400',
  },
  {
    border: 'border-amber-500/20',
    bg: 'bg-amber-500/5',
    iconBg: 'bg-amber-500/10',
    accent: 'text-amber-400',
  },
  {
    border: 'border-purple-500/20',
    bg: 'bg-purple-500/5',
    iconBg: 'bg-purple-500/10',
    accent: 'text-purple-400',
  },
  {
    border: 'border-rose-500/20',
    bg: 'bg-rose-500/5',
    iconBg: 'bg-rose-500/10',
    accent: 'text-rose-400',
  },
];

/**
 * Extracts a score impact like "+45 points" or "+15 points" from a tip string.
 */
function extractImpact(tip) {
  const match = tip.match(/\+(\d+)\s*(?:points?|pts)/i);
  return match ? `+${match[1]} pts` : null;
}

export default function CoachingTips({ tips = [] }) {
  if (tips.length === 0) {
    return (
      <div className="glass-card p-6" id="coaching-tips">
        <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider mb-4">
          🎯 Coaching Tips
        </h3>
        <p className="text-surface-500 text-sm">No coaching tips available yet.</p>
      </div>
    );
  }

  return (
    <div className="glass-card p-6" id="coaching-tips">
      <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider mb-4">
        🎯 Coaching Tips
      </h3>

      <div className="space-y-3">
        {tips.map((tip, i) => {
          const style = TIP_STYLES[i % TIP_STYLES.length];
          const icon = TIP_ICONS[i % TIP_ICONS.length];
          const impact = extractImpact(tip);

          return (
            <div
              key={i}
              className={`
                flex items-start gap-3 p-4 rounded-xl border
                ${style.border} ${style.bg}
                transition-all duration-200
                hover:translate-x-1 hover:shadow-lg
              `}
            >
              {/* Icon */}
              <span
                className={`
                  flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center
                  text-lg ${style.iconBg}
                `}
              >
                {icon}
              </span>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-surface-200 leading-relaxed">
                  {tip}
                </p>
              </div>

              {/* Score impact badge */}
              {impact && (
                <span className={`
                  flex-shrink-0 text-xs font-bold px-2 py-1 rounded-md
                  ${style.iconBg} ${style.accent}
                `}>
                  {impact}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
