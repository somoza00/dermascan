import styles from './Disclaimer.module.css';

export function Disclaimer() {
  return (
    <div className={styles.banner}>
      <div className={styles.inner}>
        <span className={styles.icon} aria-hidden>⚠️</span>
        <p>
          <strong>Importante:</strong> O DermaScan é uma ferramenta de suporte educacional e clínico. Os resultados não constituem diagnóstico médico. A imagem deve ser de qualidade dermatoscópica para resultados confiáveis.{' '}
          <strong>Sempre consulte um dermatologista</strong> para avaliação e diagnóstico definitivo.
        </p>
      </div>
    </div>
  );
}