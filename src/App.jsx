import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './views/Dashboard';
import Analysis from './views/Analysis';
import './App.css';

function App() {
  const [activeView, setActiveView] = useState('dashboard');

  const getViewTitle = () => {
    switch (activeView) {
      case 'dashboard': return 'Market Overview';
      case 'analysis': return 'Security Analysis';
      case 'portfolio-designer': return 'Portfolio Designer';
      case 'portfolio-analysis': return 'Portfolio Analysis';
      case 'comparison': return 'Portfolio Comparison';
      default: return 'InvestAI';
    }
  };

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
        return <Dashboard />;
      case 'analysis':
        return <Analysis />;
      default:
        return (
          <div className="placeholder-view">
            <h2>{getViewTitle()} Coming Soon</h2>
            <p>Our agentic AI is currently preparing the data for this module.</p>
          </div>
        );
    }
  };

  return (
    <div className="app-container">
      <Sidebar activeView={activeView} setActiveView={setActiveView} />
      <div className="main-wrapper">
        <Header title={getViewTitle()} />
        <main className="content">
          {renderView()}
        </main>
      </div>
    </div>
  );
}

export default App;
