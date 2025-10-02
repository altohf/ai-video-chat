import React, { useState, useEffect } from 'react';
import './App.css';

interface User {
  name: string;
  plan: string;
  credits: number;
}

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const handleDemoLogin = () => {
    setUser({ name: 'Demo User', plan: 'FREE', credits: 45 });
    setIsLoggedIn(true);
  };

  return (
    <div className="App">
      {/* Animated Background */}
      <div className="bg-animation">
        <div className="neural-network">
          {[...Array(50)].map((_, i) => (
            <div key={i} className={`node node-${i % 5}`} />
          ))}
        </div>
        <div className="floating-particles">
          {[...Array(20)].map((_, i) => (
            <div key={i} className={`particle particle-${i % 3}`} />
          ))}
        </div>
      </div>

      {/* Mouse Glow Effect */}
      <div 
        className="mouse-glow" 
        style={{
          left: mousePosition.x - 100,
          top: mousePosition.y - 100,
        }}
      />

      <header className="header">
        <div className="container">
          <div className="logo">
            <div className="logo-icon">
              <div className="ai-brain">
                <div className="brain-core"></div>
                <div className="brain-pulse"></div>
              </div>
            </div>
            <div className="logo-text">
              <h1>AI Video Chat</h1>
              <span className="tagline">Next-Gen Intelligence</span>
            </div>
          </div>
          
          <div className="auth-section">
            {isLoggedIn ? (
              <div className="user-info">
                <div className="user-avatar">
                  <div className="avatar-glow"></div>
                  DU
                </div>
                <div className="user-details">
                  <span className="user-name">Benvenuto, {user?.name}!</span>
                  <span className="user-plan">
                    <span className="plan-badge">{user?.plan}</span>
                    <span className="credits">{user?.credits} crediti</span>
                  </span>
                </div>
              </div>
            ) : (
              <button className="login-btn" onClick={handleDemoLogin}>
                <span className="btn-text">Accedi Demo</span>
                <div className="btn-glow"></div>
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="main">
        <div className="container">
          {!isLoggedIn ? (
            <div className="hero">
              <div className="hero-badge">
                <span className="badge-text">🚀 Powered by GPT-4 Vision</span>
                <div className="badge-glow"></div>
              </div>
              
              <h1 className="hero-title">
                Trasforma i tuoi video in
                <span className="gradient-text"> conversazioni intelligenti</span>
              </h1>
              
              <p className="hero-subtitle">
                L'intelligenza artificiale più avanzata analizza ogni frame, 
                ogni parola, ogni dettaglio dei tuoi video per offrirti 
                un'esperienza conversazionale rivoluzionaria.
              </p>

              <div className="cta-section">
                <button className="cta-primary" onClick={handleDemoLogin}>
                  <span>Inizia Gratis</span>
                  <div className="cta-arrow">→</div>
                </button>
                <button className="cta-secondary">
                  <span>Guarda Demo</span>
                  <div className="play-icon">▶</div>
                </button>
              </div>

              <div className="features-grid">
                <div className="feature-card">
                  <div className="feature-icon brain">
                    <div className="icon-core"></div>
                    <div className="icon-rings"></div>
                  </div>
                  <h3>Analisi Neurale</h3>
                  <p>Deep learning per comprensione completa di video e audio</p>
                  <div className="feature-glow"></div>
                </div>

                <div className="feature-card">
                  <div className="feature-icon chat">
                    <div className="chat-bubbles">
                      <div className="bubble"></div>
                      <div className="bubble"></div>
                      <div className="bubble"></div>
                    </div>
                  </div>
                  <h3>Chat Intelligente</h3>
                  <p>Conversazioni naturali con comprensione contestuale avanzata</p>
                  <div className="feature-glow"></div>
                </div>

                <div className="feature-card">
                  <div className="feature-icon quantum">
                    <div className="quantum-core"></div>
                    <div className="quantum-orbits"></div>
                  </div>
                  <h3>Elaborazione Quantica</h3>
                  <p>Velocità di processamento e precisione senza precedenti</p>
                  <div className="feature-glow"></div>
                </div>
              </div>
            </div>
          ) : (
            <div className="upload-section">
              <div className="upload-header">
                <h2>Carica il tuo video</h2>
                <p>L'AI analizzerà ogni dettaglio in tempo reale</p>
              </div>
              
              <div className="upload-container">
                <div className="upload-box">
                  <div className="upload-visual">
                    <div className="upload-icon">
                      <div className="upload-core"></div>
                      <div className="upload-waves"></div>
                    </div>
                  </div>
                  
                  <h3>Trascina il video qui</h3>
                  <p>Oppure clicca per selezionare</p>
                  
                  <div className="upload-specs">
                    <span>MP4, AVI, MOV, MKV, WEBM</span>
                    <span>•</span>
                    <span>Max 100MB</span>
                  </div>
                  
                  <button className="upload-btn">
                    <span>Seleziona Video</span>
                    <div className="btn-pulse"></div>
                  </button>
                </div>
                
                <div className="ai-preview">
                  <div className="ai-status">
                    <div className="status-indicator"></div>
                    <span>AI pronta per l'analisi</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
