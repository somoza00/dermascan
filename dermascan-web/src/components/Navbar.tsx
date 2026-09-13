import styles from './Navbar.module.css';

export function Navbar() {
  return (
    <nav className={styles.nav}>
      <a className={styles.logo} href="#">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8" />
          <path d="M21 21l-4.35-4.35M11 8v6M8 11h6" />
        </svg>
        DermaScan
      </a>
      <a className={styles.cta} href="#analisar">Analisar imagem</a>
    </nav>
  );
}