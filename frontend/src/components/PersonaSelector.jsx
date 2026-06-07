/**
 * PersonaSelector — Dropdown to choose between demo personas.
 *
 * Props:
 *   personas       {Array}   List of persona objects from /demo/personas
 *   selectedId     {string}  Currently selected borrower_id
 *   onSelect       {fn}      Callback with borrower_id
 *   loading        {boolean} Whether personas are being loaded
 */

import { useState } from 'react';

function getScoreColor(score) {
  if (score >= 700) return 'text-green-400';
  if (score >= 600) return 'text-yellow-400';
  return 'text-red-400';
}

function getInitials(name) {
  return name
    .split(' ')
    .map(w => w[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);
}

const PLATFORM_EMOJIS = {
  swiggy_partner: '🛵',
  upwork: '💻',
  toptal: '🔷',
  fiverr: '🎨',
  default: '📱',
};

export default function PersonaSelector({
  personas = [],
  selectedId = '',
  onSelect,
  loading = false,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const selected = personas.find(p => p.borrower_id === selectedId);

  return (
    <div className="relative" id="persona-selector">
      {/* Selected persona display */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={loading || personas.length === 0}
        className={`
          w-full flex items-center gap-3 px-4 py-3 rounded-xl
          border border-surface-600 bg-surface-800/50
          hover:border-surface-500 hover:bg-surface-700/50
          transition-all duration-200
          disabled:opacity-50 disabled:cursor-not-allowed
          focus:outline-none focus:ring-2 focus:ring-primary-500/30
        `}
      >
        {selected ? (
          <>
            {/* Avatar */}
            <div className="w-10 h-10 rounded-full bg-primary-600/20 border border-primary-500/30 flex items-center justify-center text-sm font-bold text-primary-400">
              {getInitials(selected.name)}
            </div>

            {/* Info */}
            <div className="flex-1 text-left">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-surface-100">{selected.name}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-700 text-surface-400">
                  {selected.platforms?.map(p => PLATFORM_EMOJIS[p] || PLATFORM_EMOJIS.default).join(' ')}
                  {' '}
                  {selected.platforms?.map(p => p.replace(/_/g, ' ')).join(', ')}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className={`text-xs font-mono font-semibold ${getScoreColor(selected.expected_score)}`}>
                  Expected: {selected.expected_score}
                </span>
                <span className="text-[10px] text-surface-500">
                  • {selected.expected_risk_label}
                </span>
              </div>
            </div>

            {/* Chevron */}
            <svg className={`w-4 h-4 text-surface-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </>
        ) : (
          <span className="text-sm text-surface-400">
            {loading ? 'Loading personas...' : 'Select a persona'}
          </span>
        )}
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 top-full left-0 right-0 mt-2 py-1 rounded-xl border border-surface-600 bg-surface-800/95 backdrop-blur-xl shadow-2xl">
          {personas.map((persona) => {
            const isActive = persona.borrower_id === selectedId;
            return (
              <button
                key={persona.borrower_id}
                onClick={() => {
                  onSelect(persona.borrower_id);
                  setIsOpen(false);
                }}
                className={`
                  w-full flex items-center gap-3 px-4 py-3
                  transition-all duration-150
                  ${isActive
                    ? 'bg-primary-600/10 border-l-2 border-primary-400'
                    : 'hover:bg-surface-700/50 border-l-2 border-transparent'
                  }
                `}
              >
                {/* Avatar */}
                <div className={`
                  w-9 h-9 rounded-full flex items-center justify-center text-xs font-bold
                  ${isActive
                    ? 'bg-primary-600/20 text-primary-400 border border-primary-500/30'
                    : 'bg-surface-700 text-surface-400 border border-surface-600'
                  }
                `}>
                  {getInitials(persona.name)}
                </div>

                {/* Info */}
                <div className="flex-1 text-left">
                  <div className="text-sm font-medium text-surface-200">{persona.name}</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className={`text-xs font-mono ${getScoreColor(persona.expected_score)}`}>
                      ~{persona.expected_score}
                    </span>
                    <span className="text-[10px] text-surface-500">
                      {persona.platforms?.map(p => PLATFORM_EMOJIS[p] || '📱').join(' ')}
                    </span>
                    <span className="text-[10px] text-surface-500">
                      ₹{(persona.income_current || 0).toLocaleString('en-IN')}/mo
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {/* Click-away overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
}
