import React, { useState, useEffect } from 'react';
import { Search, Loader2, TrendingUp, TrendingDown, Info, ShieldAlert, Target, CheckCircle2 } from 'lucide-react';
import './Analysis.css';

const Analysis = () => {
    const [ticker, setTicker] = useState('');
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);

    // Restore persisted analysis state (if any) when component mounts
    useEffect(() => {
        try {
            const saved = localStorage.getItem('analysisState');
            if (saved) {
                const parsed = JSON.parse(saved);
                if (parsed) {
                    setTicker(parsed.ticker || '');
                    setData(parsed.data || null);
                    setError(parsed.error || null);
                    // Keep loading false on mount; any in-progress operations won't be resumed.
                    setLoading(false);
                }
            }
        } catch (e) {
            console.warn('Failed to restore analysis state:', e);
        }
    }, []);

    // Persist relevant state whenever it changes so navigation away doesn't reset the view
    useEffect(() => {
        try {
            const payload = { ticker, data, error };
            localStorage.setItem('analysisState', JSON.stringify(payload));
        } catch (e) {
            console.warn('Failed to save analysis state:', e);
        }
    }, [ticker, data, error]);

    // Helper to render formatted analysis text
    const renderAnalysisText = (value) => {
        if (typeof value !== 'string') {
            if (typeof value === 'object' && value !== null) {
                return JSON.stringify(value, null, 2);
            }
            return 'N/A';
        }

        // Convert **bold** to <strong> tags
        let formatted = value.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert bullet points to proper list items
        const lines = formatted.split('\n');
        let inList = false;
        let result = [];

        for (let line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('•')) {
                if (!inList) {
                    result.push('<ul>');
                    inList = true;
                }
                result.push(`<li>${trimmed.substring(1).trim()}</li>`);
            } else {
                if (inList) {
                    result.push('</ul>');
                    inList = false;
                }
                if (trimmed) {
                    result.push(`<p>${trimmed}</p>`);
                }
            }
        }

        if (inList) result.push('</ul>');

        return <div dangerouslySetInnerHTML={{ __html: result.join('') }} />;
    };

    const handleSearch = async (e) => {
        e.preventDefault();
        if (!ticker) return;

        setLoading(true);
        setError(null);
        setData(null);

        try {
            console.log('Analyzing ticker:', ticker);
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: ticker.trim() }),
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error response:', errorText);
                throw new Error(`Failed to analyze security: ${response.status}`);
            }

            const result = await response.json();
            console.log('Analysis result:', result);

            // Validate the response structure
            if (!result || !result.snapshot || !result.analysis) {
                console.error('Invalid response structure:', result);
                throw new Error('Invalid response from server');
            }

            setData(result);
        } catch (err) {
            console.error('Error in handleSearch:', err);
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
                                    <div>
                                        <h3 className="text-gold">Security Snapshot</h3>
                                        <span className="badge">{data.snapshot.ticker}</span>
                                    </div>
                                    {data.snapshot.analyst_ratings?.recommendation && (
                                        <span className={`sentiment-badge ${data.snapshot.analyst_ratings.recommendation.toLowerCase()}`}>
                                            {data.snapshot.analyst_ratings.recommendation}
                                        </span>
                                    )}
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
                                    <div className="metric">
                                        <span>Analyst Target</span>
                                        <h3>{data.snapshot.analyst_ratings?.target_mean ? `${data.snapshot.analyst_ratings.target_mean} ${data.snapshot.currency}` : 'N/A'}</h3>
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
                                    <p>{renderAnalysisText(data.analysis.investment_thesis)}</p>
                                </div>

                                <div className="report-divider"></div>

                                <div className="report-section">
                                    <div className="section-header">
                                        <ShieldAlert size={18} className="text-gold" />
                                        <h4>Risk/Reward Analysis</h4>
                                    </div>
                                    <p>{renderAnalysisText(data.analysis.risk_reward)}</p>
                                </div>

                                <div className="report-divider"></div>

                                <div className="report-section">
                                    <div className="section-header">
                                        <TrendingUp size={18} className="text-gold" />
                                        <h4>Portfolio Fit</h4>
                                    </div>
                                    <p>{renderAnalysisText(data.analysis.portfolio_fit)}</p>
                                </div>

                                <div className="bottom-line">
                                    <p><strong>The Bottom Line:</strong> {renderAnalysisText(data.analysis.bottom_line)}</p>
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
