import styles from './Hero.module.css';

export function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.glow} aria-hidden />
      <div className={styles.eyebrow}>
        <span className={styles.dot} />
        Triagem inteligente com EfficientNet-B3
      </div>
      <h1 className={styles.h1}>
        Detecção precoce de<br />
        <em>lesões de pele</em> com IA
      </h1>
      <p className={styles.sub}>
        Envie uma imagem dermatoscópica e receba em segundos uma análise
        preliminar das principais categorias de lesões cutâneas. Uma ferramenta
        de suporte à decisão — não substitui o dermatologista.
      </p>
      <a className={styles.heroCta} href="#analisar">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        Enviar imagem agora
      </a>
      <div className={styles.stats}>
        <div className={styles.statItem}>
          <span className={styles.statNum}>85%</span>
          <span className={styles.statLabel}>Acurácia de validação</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statNum}>7</span>
          <span className={styles.statLabel}>Categorias de lesões</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statNum}>&lt;2s</span>
          <span className={styles.statLabel}>Tempo de análise</span>
        </div>
      </div>
    </section>
  );
}