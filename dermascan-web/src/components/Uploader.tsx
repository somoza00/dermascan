import { useRef, useState, useCallback } from 'react';
import { predict, type PredictResponse } from '../services/api';
import styles from './Uploader.module.css';

type Status = 'idle' | 'preview' | 'loading' | 'done' | 'error';

// Mapeia risk_level para a cor do diagnóstico. A API devolve os valores do
// enum `"high" | "medium" | "low"` (ver app/schemas/prediction.py); aceita
// também os rótulos em pt ("alto"/"moderado"/"baixo") como reforço defensivo.
function riskColor(riskLevel: string): string {
  const level = riskLevel.toLowerCase();
  if (level.includes('high') || level.includes('alto')) return 'var(--risk-high)';
  if (level.includes('medium') || level.includes('moderado')) return 'var(--risk-med)';
  return 'var(--risk-low)';
}

export function Uploader() {
  const inputRef              = useRef<HTMLInputElement>(null);
  const [status, setStatus]   = useState<Status>('idle');
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile]       = useState<File | null>(null);
  const [result, setResult]   = useState<PredictResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const loadFile = useCallback((f: File) => {
    if (!f.type.match(/image\/(jpeg|png)/)) { setError('Use JPEG ou PNG.'); return; }
    if (f.size > 10 * 1024 * 1024) { setError('Arquivo muito grande (máx 10 MB).'); return; }
    const reader = new FileReader();
    reader.onload = e => {
      setPreview(e.target?.result as string);
      setFile(f);
      setStatus('preview');
      setResult(null);
      setError(null);
    };
    reader.readAsDataURL(f);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
  }, [loadFile]);

  const onAnalyze = async () => {
    if (!file) return;
    setStatus('loading');
    setError(null);
    try {
      const res = await predict(file);
      setResult(res);
      setStatus('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erro desconhecido.');
      setStatus('error');
    }
  };

  const confidencePct = result ? Math.round(result.confidence * 100) : 0;

  return (
    <section className={styles.section} id="analisar">
      <div className={styles.container}>
        <p className={styles.label}>Análise de imagem</p>
        <h2 className={styles.title}>Envie sua imagem</h2>
        <p className={styles.sub}>Use uma imagem dermatoscópica nítida. Fotos comuns de câmera podem reduzir a precisão da análise.</p>

        <div className={styles.wrapper}>
          <div
            className={`${styles.dropZone} ${dragOver ? styles.dragOver : ''}`}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
            aria-label="Clique ou arraste uma imagem"
          >
            <input ref={inputRef} type="file" accept="image/jpeg,image/png" className={styles.hiddenInput}
              onChange={e => { if (e.target.files?.[0]) loadFile(e.target.files[0]); }} />
            <UploadIcon className={styles.uploadIcon} />
            <p className={styles.dropTitle}>Arraste uma imagem ou clique para selecionar</p>
            <p className={styles.dropSub}>JPEG ou PNG · Máx. 10 MB</p>
          </div>

          {(status === 'preview' || status === 'loading' || status === 'done' || status === 'error') && preview && (
            <div className={styles.previewCard}>
              <img src={preview} alt="Pré-visualização" className={styles.previewImg} />
              <button className={styles.analyzeBtn} onClick={onAnalyze} disabled={status === 'loading'}>
                {status === 'loading'
                  ? <><span className={styles.spinner} /> Analisando…</>
                  : <><SearchIcon /> Analisar imagem</>
                }
              </button>
            </div>
          )}

          {error && <div className={styles.errorBox} role="alert">{error}</div>}

          {status === 'done' && result && (
            <div className={styles.resultCard}>
              <p className={styles.resultLabel}>Classificação preliminar</p>
              <p className={styles.diagnosis} style={{ color: riskColor(result.risk_level) }}>{result.label}</p>
              <div className={styles.confWrap}>
                <div className={styles.confHeader}>
                  <span>Confiança do modelo</span>
                  <span>{confidencePct}%</span>
                </div>
                <div className={styles.confBar}>
                  <div className={styles.confFill} style={{ width: `${confidencePct}%` }} />
                </div>
              </div>
              {result.inference_mode === 'real' && (
                <p className={styles.modeBadge}>✓ Inferência real · EfficientNet-B3</p>
              )}
              <p className={styles.disclaimer}>
                Este resultado é gerado por um modelo com acurácia de validação de 85%. Não substitui avaliação dermatológica profissional. Procure um especialista para diagnóstico definitivo.
              </p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function UploadIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8.5" cy="8.5" r="1.5" /><polyline points="21 15 16 10 5 21" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
    </svg>
  );
}