import styles from './HowItWorks.module.css';

const STEPS = [
  { num: '01', title: 'Envie a imagem', desc: 'Faça upload de uma foto dermatoscópica de boa qualidade da lesão (JPEG ou PNG, máx. 10 MB).' },
  { num: '02', title: 'Processamento por IA', desc: 'O modelo EfficientNet-B3 analisa os padrões visuais da lesão e calcula a probabilidade para cada categoria.' },
  { num: '03', title: 'Resultado e orientação', desc: 'Você recebe a classificação com nível de confiança e orientação sobre a urgência de buscar avaliação médica.' },
];

export function HowItWorks() {
  return (
    <section className={styles.section}>
      <div className={styles.container}>
        <p className={styles.label}>Como funciona</p>
        <h2 className={styles.title}>Três passos para uma análise</h2>
        <p className={styles.sub}>O DermaScan usa uma rede neural treinada em milhares de imagens clínicas para identificar padrões visuais associados a lesões de pele.</p>
        <div className={styles.grid}>
          {STEPS.map(s => (
            <div key={s.num} className={styles.card}>
              <div className={styles.stepNum}>{s.num}</div>
              <h3 className={styles.stepTitle}>{s.title}</h3>
              <p className={styles.stepDesc}>{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}