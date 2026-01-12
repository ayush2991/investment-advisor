import React, { useState } from 'react';
import { Search, Loader2, TrendingUp, TrendingDown, Info, ShieldAlert, Target, CheckCircle2 } from 'lucide-react';
import './Analysis.css';

const Analysis = () => {
    const [ticker, setTicker] = useState('');
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!ticker) return;

        setLoading(true);
        setError(null);
        setData(null);

        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: ticker.trim() }),
            });

            if (!response.ok) throw new Error('Failed to analyze security');

            const result = await response.json();
            setData(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="analysis-view">
            <div className="search-container">
                <form onSubmit={handleSearch} className="analysis-search">
                    <Search className="search-icon" size={20} />
                    <input
                        type="text"
                        placeholder="Enter ticker (e.g. AAPL, TSLA, NVDA)"
                        value={ticker}
                        onChange={(e) => setTicker(e.target.value)}
                    />
                    <button type="submit" disabled={loading}>
                        {loading ? <Loader2 className="animate-spin" size={18} /> : 'Analyze'}
                    </button>
                </form>
            </div>

            {loading && (
                <div className="agent-loading">
                    <div className="loader"></div>
                    <p>Orchestrator delegating tasks to agents...</p>
                    <div className="loading-steps">
                        <span className="step active">Quantitative Agent analyzing financials...</span>
                        <span className="step">Market Intelligence gathering news...</span>
                        <span className="step">Lead Analyst synthesizing report...</span>
                    </div>
                </div>
            )}

            {error && <div className="error-message">{error}</div>}

            {data && (
                <div className="analysis-results">
                    <div className="results-grid">
                        {/* Snapshot Column */}
                        <div className="snapshot-column">
                            <div className="card snapshot-card">
                                <div className="card-header">
                                    <h3 className="text-gold">Security Snapshot</h3>
                                    <span className="badge">{data.snapshot.ticker}</span>
                                </div>
                                <div className="snapshot-metrics">
                                    <div className="metric">
                                        <span>Price</span>
                                        <h3>{data.snapshot.price} {data.snapshot.currency}</h3>
                                    </div>
                                    <div className="metric">
                                        <span>P/E Ratio</span>
                                        <h3>{data.snapshot.pe_ratio?.toFixed(2) || 'N/A'}</h3>
                                    </div>
                                    <div className="metric">
                                        <span>Debt/Equity</span>
                                        <h3>{data.snapshot.debt_to_equity?.toFixed(2) || 'N/A'}</h3>
                                    </div>
                                </div>
                            </div>

                            <div className="card sentiment-card mt-6">
                                <div className="card-header">
                                    <h3>Market Sentiment</h3>
                                </div>
                                <div className="sentiment-details">
                                    <div className="sentiment-item">
                                        <Target size={18} className="text-gold" />
                                        <div>
                                            <p className="label">Analyst Target</p>
                                            <p className="value">{data.snapshot.analyst_ratings?.target_mean || 'N/A'}</p>
                                        </div>
                                    </div>
                                    <div className="sentiment-item">
                                        <CheckCircle2 size={18} className="text-gold" />
                                        <div>
                                            <p className="label">Recommendation</p>
                                            <p className="value text-capitalize">{data.snapshot.analyst_ratings?.recommendation || 'N/A'}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Analysis Column */}
                        <div className="analysis-column">
                            <div className="card report-card">
                                <div className="report-section">
                                    <div className="section-header">
                                        <Info size={18} className="text-gold" />
                                        <h4>Investment Thesis</h4>
                                    </div>
                                    <p>{data.analysis.investment_thesis}</p>
                                </div>

                                <div className="report-divider"></div>

                                <div className="report-section">
                                    <div className="section-header">
                                        <ShieldAlert size={18} className="text-gold" />
                                        <h4>Risk/Reward Analysis</h4>
                                    </div>
                                    <p>{data.analysis.risk_reward}</p>
                                </div>

                                <div className="report-divider"></div>

                                <div className="report-section">
                                    <div className="section-header">
                                        <TrendingUp size={18} className="text-gold" />
                                        <h4>Portfolio Fit</h4>
                                    </div>
                                    <p>{data.analysis.portfolio_fit}</p>
                                </div>

                                <div className="bottom-line">
                                    <p><strong>The Bottom Line:</strong> {data.analysis.bottom_line}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Analysis;
