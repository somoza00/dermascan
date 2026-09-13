import styles from './SkinTypes.module.css';

type Risk = 'low' | 'med' | 'high';

interface SkinType {
  icon: string;
  name: string;
  risk: Risk;
  riskLabel: string;
  desc: string;
}

const TYPES: SkinType[] = [
  { icon: '🔴', name: 'Melanoma', risk: 'high', riskLabel: 'Alto risco', desc: 'O câncer de pele mais perigoso. Surge de melanócitos e pode se espalhar rapidamente. Detecção precoce é determinante para o prognóstico.' },
  { icon: '🔴', name: 'Carcinoma Basocelular', risk: 'high', riskLabel: 'Alto risco', desc: 'Tipo mais comum de câncer de pele. Raramente metastático, mas pode causar destruição local significativa se não tratado.' },
  { icon: '🔴', name: 'Lesão Vascular', risk: 'high', riskLabel: 'Alto risco', desc: 'Inclui hemangiomas e outras anomalias vasculares cutâneas. Algumas variantes requerem avaliação especializada urgente.' },
  { icon: '🟡', name: 'Queratose Actínica', risk: 'med', riskLabel: 'Risco moderado', desc: 'Lesão pré-cancerosa causada por exposição solar crônica. Pode evoluir para carcinoma espinocelular se não tratada.' },
  { icon: '🟡', name: 'Queratose Seborreica', risk: 'med', riskLabel: 'Risco moderado', desc: 'Lesão benigna comum em adultos mais velhos. Pode ser difícil de distinguir de lesões malignas visualmente.' },
  { icon: '🟢', name: 'Nevo Melanocítico', risk: 'low', riskLabel: 'Baixo risco', desc: 'Pintas comuns, geralmente benignas. Monitoramento recomendado quando há mudanças de tamanho, cor ou bordas (regra ABCDE).' },
  { icon: '🟢', name: 'Dermatofibroma', risk: 'low', riskLabel: 'Baixo risco', desc: 'Nódulo benigno de tecido fibroso, comum em membros inferiores. Raramente requer tratamento; acompanhamento periódico é suficiente.' },
];

export function SkinTypes() {
  return (
    <section className={styles.section} id="tipos">
      <div className={styles.container}>
        <p className={styles.label}>O que o modelo identifica</p>
        <h2 className={styles.title}>Categorias de lesões cutâneas</h2>
        <p className={styles.sub}>O modelo foi treinado para distinguir entre lesões benignas e malignas. Conheça as categorias e seus níveis de risco clínico.</p>
        <div className={styles.grid}>
          {TYPES.map(t => (
            <div key={t.name} className={`${styles.card} ${styles[t.risk]}`}>
              <div className={styles.icon}>{t.icon}</div>
              <div className={styles.header}>
                <span className={styles.name}>{t.name}</span>
                <span className={styles.badge}>{t.riskLabel}</span>
              </div>
              <p className={styles.desc}>{t.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}