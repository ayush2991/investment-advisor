import React from 'react';
import {
  LayoutDashboard,
  Search,
  PieChart,
  BarChart3,
  Layers,
  Settings,
  ShieldCheck,
  Zap,
  User
} from 'lucide-react';
import './Sidebar.css';

const Sidebar = ({ activeView, setActiveView }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={18} strokeWidth={1.5} /> },
    { id: 'analysis', label: 'Analysis', icon: <Search size={18} strokeWidth={1.5} /> },
    { id: 'portfolio-designer', label: 'Designer', icon: <PieChart size={18} strokeWidth={1.5} /> },
    { id: 'portfolio-analysis', label: 'Insights', icon: <BarChart3 size={18} strokeWidth={1.5} /> },
    { id: 'comparison', label: 'Compare', icon: <Layers size={18} strokeWidth={1.5} /> },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <ShieldCheck className="logo-icon" color="var(--accent)" strokeWidth={2.5} />
        <span className="logo-text">InvestAI</span>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeView === item.id ? 'active' : ''}`}
            onClick={() => setActiveView(item.id)}
          >
            {item.icon}
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button className="nav-item">
          <Settings size={18} strokeWidth={1.5} />
          <span>Configuration</span>
        </button>
        <div className="user-badge">
          <div className="user-avatar">
            <User size={14} strokeWidth={2} />
          </div>
          <div className="user-info">
            <span className="user-name">Aayush Agarwal</span>
            <span className="user-plan">Premium Member</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
