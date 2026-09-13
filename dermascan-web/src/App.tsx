import { Navbar }     from './components/Navbar';
import { Hero }       from './components/Hero';
import { HowItWorks } from './components/HowItWorks';
import { SkinTypes }  from './components/SkinTypes';
import { Disclaimer } from './components/Disclaimer';
import { Uploader }   from './components/Uploader';
import { Footer }     from './components/Footer';

export default function App() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <HowItWorks />
        <SkinTypes />
        <Disclaimer />
        <Uploader />
      </main>
      <Footer />
    </>
  );
}