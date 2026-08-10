import { useState } from 'react';
import { ImageUploader } from '../components/ImageUploader';
import { ResultCard } from '../components/ResultCard';
import { Disclaimer } from '../components/Disclaimer';
import { predictImage } from '../services/api';
import type { PredictionResult } from '../services/api';

export function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImage = async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro inesperado. Tente novamente.');
    } finally {
      setLoading(false);
    }
  };

  const handleRejected = (reason: string) => {
    setError(reason);
    setResult(null);
  };

  return (
    <div style={{ maxWidth: 520, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 32 }}>
        <div style={{ fontSize: 48, marginTop: -8, marginBottom: 20 }}>🔬</div>
        <h1 style={{ color: '#f1f5f9', margin: 0, fontSize: 28 }}>DermaScan</h1>
        <p style={{ color: '#64748b', marginTop: 8, fontSize: 15 }}>
          Análise preliminar de lesões de pele com IA
        </p>
      </div>

      {/* Upload */}
      <ImageUploader onImageSelect={handleImage} onRejected={handleRejected} disabled={loading} />

      {/* Loading */}
      {loading && (
        <div style={{ textAlign: 'center', marginTop: 24, color: '#94a3b8' }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>⏳</div>
          <p>Analisando imagem...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          marginTop: 24,
          padding: 12,
          background: '#450a0a',
          borderRadius: 8,
          color: '#fca5a5',
          border: '1px solid #ef444433',
        }}>
          ❌ {error}
        </div>
      )}

      {/* Result */}
      {result && <ResultCard result={result} />}

      {/* Disclaimer */}
      <Disclaimer />
    </div>
  );
}
