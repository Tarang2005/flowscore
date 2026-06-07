/**
 * IncomeChart — 6-month income trend line chart.
 *
 * Props:
 *   platforms  {Array}  Platform data with monthly_earnings_last_6m
 *   currency   {string} Currency symbol (default "₹")
 */

import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Area, AreaChart,
} from 'recharts';

const MONTH_LABELS = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'];

function buildChartData(platforms) {
  if (!platforms || platforms.length === 0) return [];

  const maxMonths = Math.max(...platforms.map(p => p.monthly_earnings_last_6m?.length || 0));

  return Array.from({ length: maxMonths }, (_, i) => {
    const entry = { month: MONTH_LABELS[i] || `M${i + 1}` };
    let total = 0;

    platforms.forEach((p) => {
      const val = p.monthly_earnings_last_6m?.[i] || 0;
      entry[p.platform] = val;
      total += val;
    });

    entry.total = total;
    return entry;
  });
}

function formatCurrency(value) {
  if (value >= 100000) return `₹${(value / 100000).toFixed(1)}L`;
  if (value >= 1000) return `₹${(value / 1000).toFixed(0)}K`;
  return `₹${value}`;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload) return null;

  return (
    <div className="bg-surface-800 border border-surface-600 rounded-lg px-4 py-3 shadow-xl">
      <p className="text-xs text-surface-400 mb-2 font-medium">{label}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2 text-sm">
          <span
            className="w-2 h-2 rounded-full"
            style={{ background: entry.color }}
          />
          <span className="text-surface-300 capitalize">
            {entry.name === 'total' ? 'Total' : entry.name.replace(/_/g, ' ')}:
          </span>
          <span className="font-semibold text-surface-100">
            ₹{entry.value?.toLocaleString('en-IN')}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function IncomeChart({ platforms = [], currency = '₹' }) {
  const data = buildChartData(platforms);

  if (data.length === 0) {
    return (
      <div className="glass-card p-6" id="income-chart">
        <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider mb-4">
          📈 Income Trend
        </h3>
        <div className="h-48 flex items-center justify-center text-surface-500 text-sm">
          No income data available
        </div>
      </div>
    );
  }

  // Compute trend percentage
  const firstTotal = data[0]?.total || 0;
  const lastTotal = data[data.length - 1]?.total || 0;
  const trendPct = firstTotal > 0 ? ((lastTotal - firstTotal) / firstTotal * 100).toFixed(0) : 0;

  return (
    <div className="glass-card p-6" id="income-chart">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-surface-300 uppercase tracking-wider">
          📈 Income Trend
        </h3>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
          trendPct > 0
            ? 'bg-green-500/10 text-green-400 border border-green-500/20'
            : 'bg-red-500/10 text-red-400 border border-red-500/20'
        }`}>
          {trendPct > 0 ? '+' : ''}{trendPct}% over 6 months
        </span>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="totalGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b7ef2" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b7ef2" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis
            dataKey="month"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={formatCurrency}
            width={55}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="total"
            stroke="#3b7ef2"
            strokeWidth={2.5}
            fill="url(#totalGradient)"
            dot={{ fill: '#3b7ef2', strokeWidth: 0, r: 4 }}
            activeDot={{ r: 6, fill: '#3b7ef2', stroke: '#fff', strokeWidth: 2 }}
          />
          {platforms.length > 1 && platforms.map((p, i) => (
            <Line
              key={p.platform}
              type="monotone"
              dataKey={p.platform}
              stroke={i === 0 ? '#8b5cf6' : '#f59e0b'}
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
