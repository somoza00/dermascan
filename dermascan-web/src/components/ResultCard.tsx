import { RiskBadge } from './RiskBadge';
import type { PredictionResult } from '../services/api';

interface ResultCardProps {
  result: PredictionResult;
}

export function ResultCard({ result }: ResultCardProps) {
  return (
    <div
      style={{
        background: '#1e293b',
        borderRadius: 16,
        padding: 24,
        marginTop: 24,
        border: '1px solid #334155',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h3 style={{ margin: 0, color: '#f1f5f9' }}>Resultado da Análise</h3>
        <RiskBadge level={result.risk_level} />
      </div>

      <div style={{ display: 'grid', gap: 12 }}>
        <div>
          <span style={{ color: '#64748b', fontSize: 14 }}>Classificação</span>
          <p style={{ color: '#f1f5f9', fontSize: 18, fontWeight: 600, margin: '4px 0' }}>
            {result.label}
          </p>
        </div>

        <div>
          <span style={{ color: '#64748b', fontSize: 14 }}>Confiança</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 4 }}>
            <div
              style={{
                flex: 1,
                height: 8,
                background: '#334155',
                borderRadius: 4,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${(result.confidence * 100).toFixed(0)}%`,
                  height: '100%',
                  background: result.confidence > 0.9 ? '#22c55e' : result.confidence > 0.7 ? '#f59e0b' : '#ef4444',
                  borderRadius: 4,
                  transition: 'width 0.5s',
                }}
              />
            </div>
            <span style={{ color: '#f1f5f9', fontWeight: 600, minWidth: 48 }}>
              {(result.confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div style={{
          marginTop: 8,
          padding: 12,
          background: '#0f172a',
          borderRadius: 8,
          borderLeft: `4px solid ${
            result.risk_level === 'high' ? '#ef4444' : result.risk_level === 'medium' ? '#f59e0b' : '#22c55e'
          }`,
        }}>
          <span style={{ color: '#64748b', fontSize: 14 }}>Recomendação</span>
          <p style={{ color: '#f1f5f9', margin: '4px 0 0' }}>{result.recommendation}</p>
        </div>
      </div>
    </div>
  );
}
