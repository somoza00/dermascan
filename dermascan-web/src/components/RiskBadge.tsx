import type { RiskLevel } from '../services/api';

interface RiskBadgeProps {
  level: RiskLevel;
}

const config: Record<RiskLevel, { label: string; color: string; bg: string }> = {
  high: { label: 'Alto Risco', color: '#ef4444', bg: '#450a0a' },
  medium: { label: 'Risco Moderado', color: '#f59e0b', bg: '#451a03' },
  low: { label: 'Baixo Risco', color: '#22c55e', bg: '#052e16' },
};

// Fallback defensivo: se a API um dia devolver um risk_level fora do union
// (ex.: divergência durante a troca mock -> modelo real), a UI degrada com
// um badge neutro em vez de lançar TypeError em render.
const FALLBACK = { label: 'Risco Indeterminado', color: '#94a3b8', bg: '#1e293b' };

export function RiskBadge({ level }: RiskBadgeProps) {
  const c = config[level] ?? FALLBACK;
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '6px 16px',
        borderRadius: 999,
        fontSize: 14,
        fontWeight: 700,
        color: c.color,
        background: c.bg,
        border: `1px solid ${c.color}33`,
      }}
    >
      {c.label}
    </span>
  );
}
