/**
 * ScoreGauge — Circular gauge displaying the FlowScore (300–850).
 * Refactored for "AI + Trust" Design System
 */

import { useEffect, useRef, useState } from 'react';

const SCORE_MIN = 300;
const SCORE_MAX = 850;
const RADIUS = 90;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function getRiskColor(risk) {
  switch(risk) {
    case 'low': return '#4FA982'; // Mint
    case 'medium': return '#eab308'; // Yellow
    case 'high': return '#f97316'; // Orange
    case 'very_high': return '#E06B6B'; // Coral
    default: return '#94a3b8';
  }
}

export default function ScoreGauge({ score = 0, riskCategory = '' }) {
  const [animatedScore, setAnimatedScore] = useState(SCORE_MIN);
  const rafRef = useRef(null);

  useEffect(() => {
    const target = Math.max(SCORE_MIN, Math.min(SCORE_MAX, score));
    const start = SCORE_MIN;
    const duration = 1200;
    const startTime = performance.now();

    function animate(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
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
  const riskColor = getRiskColor(riskCategory);

  return (
    <div className="glass-card p-6 flex flex-col relative overflow-hidden" id="score-gauge">
      {/* Title */}
      <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-widest z-10 mb-4">
        Credit Tracker
      </h3>

      {/* Ambient Fog / AI Glow */}
      <div 
        className="absolute inset-0 z-0 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at center, rgba(139, 92, 246, 0.08) 0%, transparent 60%)',
          filter: 'blur(40px)',
          transform: 'translateY(10%)'
        }}
      />

      <div className="flex flex-col items-center justify-center relative z-10 my-4">
        {/* SVG Gauge */}
        <div className="relative w-64 h-64">
          <svg viewBox="0 0 200 200" className="w-full h-full -rotate-135 drop-shadow-[0_0_15px_rgba(139,92,246,0.3)]">
            <defs>
              <linearGradient id="filamentGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="var(--color-violet)" />
                <stop offset="100%" stopColor="var(--color-magenta)" />
              </linearGradient>
            </defs>
            {/* Background track */}
            <circle
              cx="100" cy="100" r={RADIUS}
              fill="none"
              stroke="rgba(255,255,255,0.04)"
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={CIRCUMFERENCE * 0.25}
            />
            {/* Score arc */}
            <circle
              cx="100" cy="100" r={RADIUS}
              fill="none"
              stroke="url(#filamentGradient)"
              strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={CIRCUMFERENCE}
              strokeDashoffset={strokeOffset}
              style={{
                transition: 'stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1)',
              }}
            />
          </svg>

          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-[64px] font-medium tracking-tight text-white leading-none">
              {animatedScore}
            </span>
            <span className="text-[11px] font-medium text-surface-400 mt-2 bg-[rgba(255,255,255,0.03)] px-3 py-1 rounded-md border border-[rgba(255,255,255,0.05)]">
              Default Risk: {riskCategory === 'very_high' ? '92.2%' : riskCategory === 'high' ? '65.4%' : riskCategory === 'medium' ? '12.8%' : '0.9%'}
            </span>
          </div>
        </div>

        {/* Risk label */}
        <div className="mt-8">
          <span
            className="px-4 py-1.5 rounded-full text-[10px] font-semibold uppercase tracking-widest flex items-center gap-2"
            style={{
              background: `${riskColor}15`,
              color: riskColor,
              border: `1px solid ${riskColor}30`,
            }}
          >
            {riskCategory === 'very_high' || riskCategory === 'high' ? '⚠️' : '✅'}
            {riskCategory.replace('_', ' ')} RISK
          </span>
        </div>
      </div>
    </div>
  );
}
