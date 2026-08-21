const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const REQUEST_TIMEOUT_MS = 30_000;
// Segurança por padrão: uma API em modo mock nunca deve parecer um produto
// clínico. Demos precisam habilitar isso deliberadamente no build.
const ALLOW_MOCK_RESULTS = import.meta.env.VITE_ALLOW_MOCK_RESULTS === 'true';

export type RiskLevel = 'high' | 'medium' | 'low';
export type InferenceMode = 'real' | 'mock';

export interface PredictionResult {
  risk_level: RiskLevel;
  label: string;
  confidence: number;
  recommendation: string;
  inference_mode: InferenceMode;
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

  const result: PredictionResult = await response.json();
  if (result.inference_mode === 'mock' && !ALLOW_MOCK_RESULTS) {
    throw new Error(
      'O servidor está em modo de demonstração e geraria um resultado aleatório. ' +
      'A exibição foi bloqueada por segurança.'
    );
  }
  return result;
}
