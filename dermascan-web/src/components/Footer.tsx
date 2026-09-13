import styles from './Footer.module.css';

export function Footer() {
  return (
    <footer className={styles.footer}>
      <p>DermaScan · Projeto de portfólio em IA · Modelo EfficientNet-B3</p>
      <p>Desenvolvido por JP · Fins educacionais</p>
    </footer>
  );
}