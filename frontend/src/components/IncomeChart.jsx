/**
 * IncomeChart — 6-month income trend line chart.
 * Refactored for "AI + Trust" Design System
 */

import {
  XAxis, Tooltip, ResponsiveContainer, Area, AreaChart,
} from 'recharts';

const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];

function buildChartData(platforms) {
  if (!platforms || platforms.length === 0) return [];
  const maxMonths = Math.max(...platforms.map(p => p.monthly_earnings_last_6m?.length || 0));

  return Array.from({ length: maxMonths }, (_, i) => {
    const entry = { month: MONTH_LABELS[i] || `M${i + 1}` };
    let total = 0;
    platforms.forEach((p) => {
      total += p.monthly_earnings_last_6m?.[i] || 0;
    });
    entry.total = total;
    return entry;
  });
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="bg-[#050810] border border-[rgba(255,255,255,0.05)] rounded-lg px-4 py-3 shadow-xl">
      <p className="text-xs text-surface-400 mb-1 font-medium">{label}</p>
      <div className="flex items-center gap-2 text-sm">
        <span className="font-semibold text-white">
          ₹{payload[0].value?.toLocaleString('en-IN')}
        </span>
      </div>
    </div>
  );
}

export default function IncomeChart({ platforms = [] }) {
  const data = buildChartData(platforms);

  if (data.length === 0) {
    return (
      <div className="glass-card p-6 h-[280px]" id="income-chart">
        <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-widest mb-4">
          Income Trend (6 Months)
        </h3>
        <div className="h-full flex items-center justify-center text-surface-500 text-sm">
          No income data available
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-6 h-[280px] flex flex-col" id="income-chart">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-semibold text-surface-400 uppercase tracking-widest z-10">
          Income Trend (6 Months)
        </h3>
      </div>

      <div className="flex-1 w-full min-h-0 relative -ml-2 -mb-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 20, right: 10, left: 10, bottom: 0 }}>
            <defs>
              <linearGradient id="lineGradient" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="var(--color-violet)" />
                <stop offset="100%" stopColor="var(--color-magenta)" />
              </linearGradient>
              <linearGradient id="fillGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--color-magenta)" stopOpacity={0.25} />
                <stop offset="100%" stopColor="var(--color-violet)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="month"
              tick={{ fill: '#64748b', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              dy={10}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.05)', strokeWidth: 1, strokeDasharray: '4 4' }} />
            <Area
              type="monotone"
              dataKey="total"
              stroke="url(#lineGradient)"
              strokeWidth={3}
              fill="url(#fillGradient)"
              dot={false}
              activeDot={{ r: 5, fill: 'var(--color-magenta)', stroke: '#fff', strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
