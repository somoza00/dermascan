import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

// Rede de segurança para respostas inesperadas da API (ex.: formato mudou
// quando o modelo real substituiu o mock). Sem isso, um valor fora do
// esperado em qualquer componente derruba a árvore inteira em tela branca.
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('DermaScan: erro inesperado na interface.', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ margin: '40px auto', maxWidth: 480, padding: 24, textAlign: 'center', color: '#f1f5f9' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>⚠️</div>
          <h2 style={{ margin: '0 0 8px' }}>Algo deu errado</h2>
          <p style={{ color: '#94a3b8', marginBottom: 16 }}>
            Não foi possível exibir o resultado da análise. Tente recarregar a página.
          </p>
          <button
            onClick={() => window.location.reload()}
            style={{
              padding: '10px 20px',
              borderRadius: 8,
              border: 'none',
              background: '#3b82f6',
              color: '#fff',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            Recarregar página
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
