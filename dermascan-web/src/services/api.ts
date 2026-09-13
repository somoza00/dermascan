const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export interface PredictResponse {
  label: string;
  confidence: number;       // 0–1
  inference_mode: 'real' | 'mock';
  probabilities?: Record<string, number>;
}

export async function predict(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API error ${res.status}: ${detail}`);
  }

  return res.json() as Promise<PredictResponse>;
}