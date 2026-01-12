import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Activity, Briefcase, AlertCircle } from 'lucide-react';
import './Dashboard.css';

const Dashboard = () => {
    const stats = [
        { label: 'EQUITY VALUE', value: '$124,592.00', change: '+2.4%', icon: <Briefcase size={16} strokeWidth={1.5} />, trend: 'up' },
        { label: 'CALCULATED P/L', value: '+$1,240.50', change: '+1.0%', icon: <TrendingUp size={16} strokeWidth={1.5} />, trend: 'up' },
        { label: 'OPEN POSITIONS', value: '18', change: '0', icon: <Activity size={16} strokeWidth={1.5} />, trend: 'neutral' },
        { label: 'BUYING POWER', value: '$12,450.25', change: '-4.2%', icon: <DollarSign size={16} strokeWidth={1.5} />, trend: 'down' },
    ];

    const news = [
        { id: 1, title: 'Federal Reserve hints at interest rate cuts in Q3', category: 'Macro', time: '2h ago', sentiment: 'positive' },
        { id: 2, title: 'Tech earnings season: What to expect from big players', category: 'Earnings', time: '4h ago', sentiment: 'neutral' },
        { id: 3, title: 'Energy stocks surge amid supply chain disruptions', category: 'Energy', time: '6h ago', sentiment: 'positive' },
        { id: 4, title: 'Regulatory changes impact fintech sector in Europe', category: 'Fintech', time: '8h ago', sentiment: 'negative' },
    ];

    return (
        <div className="dashboard">
            <div className="dashboard-grid">
                {stats.map((stat, index) => (
                    <div key={index} className="stat-card">
                        <div className="stat-header">
                            <div className="stat-icon">{stat.icon}</div>
                            <span className={`stat-change ${stat.trend}`}>{stat.change}</span>
                        </div>
                        <div className="stat-body">
                            <h3>{stat.value}</h3>
                            <p>{stat.label}</p>
                        </div>
                    </div>
                ))}
            </div>

            <div className="dashboard-layout mt-8">
                <div className="main-dashboard-content">
                    <div className="card performance-card">
                        <div className="card-header">
                            <h2>Portfolio Performance</h2>
                            <div className="card-actions">
                                <button className="btn-sm active">1D</button>
                                <button className="btn-sm">1W</button>
                                <button className="btn-sm">1M</button>
                                <button className="btn-sm">YTD</button>
                                <button className="btn-sm">ALL</button>
                            </div>
                        </div>
                        <div className="chart-placeholder">
                            <div className="mock-chart">
                                {/* Mock data bars for visual flair */}
                                {[...Array(20)].map((_, i) => (
                                    <div
                                        key={i}
                                        className="mock-bar"
                                        style={{ height: `${20 + Math.random() * 60}%`, opacity: 0.3 + (i / 20) }}
                                    ></div>
                                ))}
                            </div>
                            <p className="chart-date">Jan 01 - Jan 11, 2026</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-6 mt-6">
                        <div className="card">
                            <div className="card-header">
                                <h2>AI Recommended Actions</h2>
                            </div>
                            <div className="action-list">
                                <div className="action-item">
                                    <div className="action-icon warning"><AlertCircle size={18} strokeWidth={1.5} /></div>
                                    <div className="action-details">
                                        <p className="action-title">Rebalance Overweight Position</p>
                                        <p className="action-desc">AAPL currently exceeds target by 4%</p>
                                    </div>
                                </div>
                                <div className="action-item">
                                    <div className="action-icon info"><TrendingUp size={18} strokeWidth={1.5} /></div>
                                    <div className="action-details">
                                        <p className="action-title">Potential Buy Signal</p>
                                        <p className="action-desc">NVDA showing strong momentum relative to sector</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="card">
                            <div className="card-header">
                                <h2>Top Movers</h2>
                            </div>
                            <div className="movers-list">
                                <div className="mover-item">
                                    <span className="ticker">TSLA</span>
                                    <span className="price">$245.12</span>
                                    <span className="change up">+5.2%</span>
                                </div>
                                <div className="mover-item">
                                    <span className="ticker">MSFT</span>
                                    <span className="price">$420.55</span>
                                    <span className="change up">+1.8%</span>
                                </div>
                                <div className="mover-item">
                                    <span className="ticker">DIS</span>
                                    <span className="price">$102.30</span>
                                    <span className="change down">-2.4%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="side-dashboard-content">
                    <div className="card news-card">
                        <div className="card-header">
                            <h2>Market Intelligence</h2>
                        </div>
                        <div className="news-list">
                            {news.map(item => (
                                <div key={item.id} className="news-item">
                                    <div className="news-meta">
                                        <span className="news-category">{item.category}</span>
                                        <span className="news-time">{item.time}</span>
                                    </div>
                                    <p className="news-title">{item.title}</p>
                                    <span className={`news-sentiment ${item.sentiment}`}>{item.sentiment}</span>
                                </div>
                            ))}
                        </div>
                        <button className="btn-link mt-4">View all news</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
