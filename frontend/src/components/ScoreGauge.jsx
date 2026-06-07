/**
 * ScoreGauge — Circular gauge displaying the FlowScore (300–850).
 *
 * Props:
 *   score        {number}  Current FlowScore (300–850)
 *   prevScore    {number}  Previous month's score (for delta arrow)
 *   riskCategory {string}  "low" | "medium" | "high" | "very_high"
 */

import { useEffect, useRef, useState } from 'react';

const SCORE_MIN = 300;
const SCORE_MAX = 850;
const RADIUS = 90;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function getScoreColor(score) {
  if (score >= 700) return { main: '#22c55e', glow: 'rgba(34, 197, 94, 0.3)', label: 'Excellent' };
  if (score >= 600) return { main: '#eab308', glow: 'rgba(234, 179, 8, 0.3)', label: 'Good' };
  if (score >= 500) return { main: '#f97316', glow: 'rgba(249, 115, 22, 0.3)', label: 'Fair' };
  return { main: '#ef4444', glow: 'rgba(239, 68, 68, 0.3)', label: 'Poor' };
}

export default function ScoreGauge({ score = 0, prevScore = null, riskCategory = '' }) {
  const [animatedScore, setAnimatedScore] = useState(SCORE_MIN);
  const rafRef = useRef(null);

  // Animate score counting up
  useEffect(() => {
    const target = Math.max(SCORE_MIN, Math.min(SCORE_MAX, score));
    const start = SCORE_MIN;
    const duration = 1200;
    const startTime = performance.now();

    function animate(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setAnimatedScore(Math.round(start + (target - start) * eased));

      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      }
    }

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [score]);

  const pct = (animatedScore - SCORE_MIN) / (SCORE_MAX - SCORE_MIN);
  const strokeOffset = CIRCUMFERENCE * (1 - pct * 0.75); // 270° arc
  const colors = getScoreColor(score);
  const delta = prevScore != null ? score - prevScore : null;

  return (
    <div className="glass-card p-6 flex flex-col items-center justify-center" id="score-gauge">
      {/* SVG Gauge */}
      <div className="relative w-56 h-56">
        <svg viewBox="0 0 200 200" className="w-full h-full -rotate-135">
          {/* Background track */}
          <circle
            cx="100" cy="100" r={RADIUS}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={CIRCUMFERENCE * 0.25}
          />
          {/* Score arc */}
          <circle
            cx="100" cy="100" r={RADIUS}
            fill="none"
            stroke={colors.main}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={strokeOffset}
            style={{
              transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)',
              filter: `drop-shadow(0 0 8px ${colors.glow})`,
            }}
          />
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-5xl font-extrabold tracking-tight"
            style={{ color: colors.main }}
          >
            {animatedScore}
          </span>
          <span className="text-xs font-medium text-surface-400 mt-1 uppercase tracking-widest">
            FlowScore
          </span>
        </div>
      </div>

      {/* Risk label + delta */}
      <div className="mt-4 flex items-center gap-3">
        <span
          className="px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider"
          style={{
            background: `${colors.main}20`,
            color: colors.main,
            border: `1px solid ${colors.main}40`,
          }}
        >
          {colors.label}
        </span>

        {delta != null && (
          <span className={`flex items-center gap-1 text-sm font-medium ${
            delta > 0 ? 'text-green-400' : delta < 0 ? 'text-red-400' : 'text-surface-400'
          }`}>
            {delta > 0 ? '▲' : delta < 0 ? '▼' : '—'}
            {Math.abs(delta)} pts
          </span>
        )}
      </div>

      {/* Score range labels */}
      <div className="mt-3 w-full flex justify-between text-[10px] text-surface-500 px-2">
        <span>300</span>
        <span>500</span>
        <span>700</span>
        <span>850</span>
      </div>
    </div>
  );
}
