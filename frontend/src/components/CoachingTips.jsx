/**
 * CoachingTips — Actionable coaching cards for borrowers.
 * Refactored for "AI + Trust" Design System
 */

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
        <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-widest mb-4">
          AI Coaching Action Items
        </h3>
        <p className="text-surface-500 text-sm">No coaching tips available yet.</p>
      </div>
    );
  }

  return (
    <div className="glass-card p-6" id="coaching-tips">
      <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-widest mb-4">
        AI Coaching Action Items
      </h3>

      <div className="flex flex-col gap-5">
        {tips.map((tip, i) => {
          const impact = extractImpact(tip);

          return (
            <div
              key={i}
              className="p-4 rounded-xl bg-[rgba(124,58,237,0.03)] transition-all duration-200 hover:bg-[rgba(124,58,237,0.06)]"
            >
              {/* Score impact badge (if exists) */}
              {impact && (
                <div className="mb-2">
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded text-white bg-[var(--color-violet)]">
                    {impact}
                  </span>
                </div>
              )}

              {/* Content */}
              <p className="text-[13px] text-surface-300 leading-relaxed">
                {tip.replace(/\+\d+\s*(?:points?|pts)\b/i, '').replace(/increase your score by/i, '').trim()}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
