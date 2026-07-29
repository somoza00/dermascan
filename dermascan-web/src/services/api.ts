const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const REQUEST_TIMEOUT_MS = 30_000;

export type RiskLevel = 'high' | 'medium' | 'low';

export interface PredictionResult {
  risk_level: RiskLevel;
  label: string;
  confidence: number;
  recommendation: string;
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data?.detail === 'string') return data.detail;
  } catch {
    // Corpo do erro não era JSON (ex.: 502/504 de um proxy/gateway) — usa o fallback abaixo.
  }
  return `Erro ao processar a imagem (código ${response.status}).`;
}

export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error('A análise demorou demais para responder. Tente novamente.');
    }
    throw new Error('Não foi possível conectar ao servidor. Verifique sua conexão e tente novamente.');
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  return response.json();
}
