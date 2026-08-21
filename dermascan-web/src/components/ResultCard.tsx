import { RiskBadge } from './RiskBadge';
import type { PredictionResult } from '../services/api';

interface ResultCardProps {
  result: PredictionResult;
}

export function ResultCard({ result }: ResultCardProps) {
  const isMock = result.inference_mode === 'mock';

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
      {isMock && (
        <div
          role="alert"
          style={{
            marginBottom: 16,
            padding: 12,
            borderRadius: 8,
            background: '#451a03',
            border: '1px solid #f59e0b',
            color: '#fde68a',
            lineHeight: 1.45,
          }}
        >
          ⚠️ <strong>Modo de demonstração:</strong> este resultado é simulado e aleatório.
          Não foi produzido por um modelo de IA e não deve orientar decisões de saúde.
        </div>
      )}
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
          <span style={{ color: '#64748b', fontSize: 14 }}>Confiança do classificador</span>
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
      {!isMock && (
        <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.4, margin: '16px 0 0' }}>
          A confiança é uma pontuação técnica do classificador, não uma probabilidade de diagnóstico.
        </p>
      )}
    </div>
  );
}
