import { Home } from './pages/Home'
import { ErrorBoundary } from './components/ErrorBoundary'
import './App.css'

function App() {
  return (
    <div className="app">
      <ErrorBoundary>
        <Home />
      </ErrorBoundary>
    </div>
  )
}

export default App
