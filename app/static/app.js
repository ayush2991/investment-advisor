if (window.__atlasInitialized) {
  console.warn("PortfolioPulse UI already initialized.");
} else {
  window.__atlasInitialized = true;
}

const form = document.getElementById("analyze-form");
const tickerInput = document.getElementById("ticker-input");
const statusBadge = document.getElementById("results-status");
const titleEl = document.getElementById("results-title");
const subtitleEl = document.getElementById("results-subtitle");
const snapshotEl = document.getElementById("snapshot");
const momentumEl = document.getElementById("momentum");
const insightsEl = document.getElementById("insights");
const newsEl = document.getElementById("news");
const portfolioForm = document.getElementById("portfolio-form");
const portfolioHoldings = document.getElementById("portfolio-holdings");
const portfolioStatus = document.getElementById("portfolio-status");
const portfolioStatsA = document.getElementById("portfolio-stats-a");
const portfolioStatsB = document.getElementById("portfolio-stats-b");
const portfolioAiContent = document.getElementById("portfolio-ai-content");
const compareForm = document.getElementById("compare-form");
const compareAInput = document.getElementById("compare-a");
const compareBInput = document.getElementById("compare-b");
const compareStatus = document.getElementById("compare-status");
const compareStatsA = document.getElementById("compare-stats-a");
const compareStatsB = document.getElementById("compare-stats-b");
const compareAiContent = document.getElementById("compare-ai-content");
const statusTitle = document.getElementById("status-title");
const statusSubtitle = document.getElementById("status-subtitle");
const statusProgress = document.getElementById("status-progress");
const cfgLookback = document.getElementById("cfg-lookback");
const cfgTradingDays = document.getElementById("cfg-trading-days");
const cfgRiskFree = document.getElementById("cfg-risk-free");
const cfgInterval = document.getElementById("cfg-interval");
const cfgAutoAdjust = document.getElementById("cfg-auto-adjust");
const cfgShortMa = document.getElementById("cfg-short-ma");
const cfgLongMa = document.getElementById("cfg-long-ma");
const cfgBenchmark = document.getElementById("cfg-benchmark");
const cfgReturnMethod = document.getElementById("cfg-return-method");
const cfgDrawdownMethod = document.getElementById("cfg-drawdown-method");
const cfgDrawdownWindow = document.getElementById("cfg-drawdown-window");
const cfgBetaLookback = document.getElementById("cfg-beta-lookback");
const cfgRiskFreeSource = document.getElementById("cfg-risk-free-source");
const cfgRiskFreeTicker = document.getElementById("cfg-risk-free-ticker");
const cfgModel = document.getElementById("cfg-model");

const tabButtons = document.querySelectorAll(".tab-button");
const tabSections = document.querySelectorAll("[data-section]");

const fmtPct = (value) => {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(2)}%`;
};

const fmtFloat = (value, digits = 2) => {
  if (value === null || value === undefined) return "—";
  return Number(value).toFixed(digits);
};

const setStatus = (state, text) => {
  statusBadge.textContent = text;
  statusBadge.style.background =
    state === "loading"
      ? "rgba(154, 208, 255, 0.18)"
      : state === "error"
        ? "rgba(255, 138, 122, 0.2)"
        : "rgba(255, 211, 139, 0.18)";
  statusBadge.style.color =
    state === "loading"
      ? "#9ad0ff"
      : state === "error"
        ? "#ff8a7a"
        : "#ffd38b";
};

const setPortfolioStatus = (state, text) => {
  portfolioStatus.textContent = text;
  portfolioStatus.style.background =
    state === "loading"
      ? "rgba(154, 208, 255, 0.18)"
      : state === "error"
        ? "rgba(255, 138, 122, 0.2)"
        : "rgba(255, 211, 139, 0.18)";
  portfolioStatus.style.color =
    state === "loading"
      ? "#9ad0ff"
      : state === "error"
        ? "#ff8a7a"
        : "#ffd38b";
};

const setCompareStatus = (state, text) => {
  compareStatus.textContent = text;
  compareStatus.style.background =
    state === "loading"
      ? "rgba(185, 167, 255, 0.18)"
      : state === "error"
        ? "rgba(255, 138, 122, 0.2)"
        : "rgba(255, 211, 139, 0.18)";
  compareStatus.style.color =
    state === "loading"
      ? "#b9a7ff"
      : state === "error"
        ? "#ff8a7a"
        : "#ffd38b";
};

const setGlobalStatus = (mode, text, active) => {
  statusTitle.textContent = mode;
  statusSubtitle.textContent = text;
  if (active) {
    statusProgress.parentElement.classList.add("active");
  } else {
    statusProgress.parentElement.classList.remove("active");
  }
};

const renderStockData = (data, fallbackTicker) => {
  titleEl.textContent = `${data.metrics.company_name || fallbackTicker} (${data.ticker})`;
  subtitleEl.textContent = `As of ${data.as_of}`;

  const snapshot = document.createDocumentFragment();
  snapshot.appendChild(buildRow("Sector", data.metrics.sector || "—"));
  snapshot.appendChild(buildRow("Industry", data.metrics.industry || "—"));
  snapshot.appendChild(buildRow("Market Cap", data.metrics.market_cap || "—"));
  snapshot.appendChild(buildRow("Price", fmtFloat(data.metrics.price)));
  snapshot.appendChild(buildRow("P/E", fmtFloat(data.metrics.pe_ratio)));
  snapshot.appendChild(buildRow("Forward P/E", fmtFloat(data.metrics.forward_pe)));
  const dividendYield =
    data.metrics.dividend_yield === null || data.metrics.dividend_yield === undefined
      ? null
      : data.metrics.dividend_yield * 100;
  const revenueGrowth =
    data.metrics.revenue_growth === null || data.metrics.revenue_growth === undefined
      ? null
      : data.metrics.revenue_growth * 100;
  snapshot.appendChild(buildRow("Dividend Yield", fmtPct(dividendYield)));
  snapshot.appendChild(buildRow("Revenue Growth", fmtPct(revenueGrowth)));

  const momentum = document.createDocumentFragment();
  momentum.appendChild(buildRow("Trend", data.metrics.trend || "—"));
  momentum.appendChild(buildRow("1Y Change", fmtPct(data.metrics.price_change_1y)));
  momentum.appendChild(buildRow("Volatility", fmtPct(data.metrics.volatility)));
  momentum.appendChild(buildRow("Max Drawdown", fmtPct(data.metrics.max_drawdown)));
  momentum.appendChild(buildRow("52W Low", fmtFloat(data.metrics.fifty_two_week_low)));
  momentum.appendChild(buildRow("52W High", fmtFloat(data.metrics.fifty_two_week_high)));
  momentum.appendChild(buildRow("Price vs 52W", fmtPct(data.metrics.price_vs_52w)));
  momentum.appendChild(buildRow("Beta", fmtFloat(data.metrics.beta)));

  snapshotEl.replaceChildren(snapshot);
  momentumEl.replaceChildren(momentum);
  insightsEl.replaceChildren(renderInsights(data.ai_summary));

  const newsContent = renderNews(data.news);
  if (typeof newsContent === "string") {
    newsEl.textContent = newsContent;
  } else {
    newsEl.replaceChildren(newsContent);
  }
};

const renderPortfolioData = (data) => {
  const countA = data.holdings ? data.holdings.length : 0;
  const countB = data.suggested_holdings ? data.suggested_holdings.length : 0;
  const maxHoldings = Math.max(countA, countB);

  const renderPortfolioOverview = (target, stats, holdings, minRows) => {
    const fragment = document.createDocumentFragment();

    // 1. Holdings first (padded)
    const hList = renderHoldingsList(holdings, minRows);
    if (typeof hList === "string") {
      fragment.append(hList);
    } else {
      fragment.appendChild(hList);
    }

    // 2. Clear Divider
    const hr = document.createElement("hr");
    hr.style.margin = "20px 0";
    hr.style.border = "none";
    hr.style.borderTop = "1px solid rgba(255, 255, 255, 0.12)";
    fragment.appendChild(hr);

    // 3. Stats second
    const statsContainer = document.createElement("div");
    statsContainer.style.display = "grid";
    statsContainer.style.gap = "8px";
    statsContainer.appendChild(buildRow("Annual Return", fmtPct(stats.annualized_return)));
    statsContainer.appendChild(buildRow("Volatility", fmtPct(stats.annualized_volatility)));
    statsContainer.appendChild(buildRow("Max Drawdown", fmtPct(stats.max_drawdown)));
    statsContainer.appendChild(buildRow("Sharpe Ratio", fmtFloat(stats.sharpe_ratio, 2)));
    statsContainer.appendChild(buildRow("Beta", fmtFloat(stats.beta, 2)));
    statsContainer.appendChild(buildRow("Effective Holdings", fmtFloat(stats.effective_holdings, 1)));
    fragment.appendChild(statsContainer);

    target.replaceChildren(fragment);
  };

  renderPortfolioOverview(portfolioStatsA, data.stats, data.holdings, maxHoldings);

  if (data.suggested_portfolio && data.suggested_portfolio.stats && portfolioStatsB) {
    renderPortfolioOverview(portfolioStatsB, data.suggested_portfolio.stats, data.suggested_holdings || [], maxHoldings);
  }

  portfolioAiContent.replaceChildren(renderPortfolioInsights(data.ai_summary));
};

const setActiveTab = (tab) => {
  tabButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  tabSections.forEach((section) => {
    const show = section.dataset.section === tab;
    if (show) {
      section.removeAttribute("hidden");
    } else {
      section.setAttribute("hidden", "true");
    }
  });
  if (tab === "stock" && window.stockLastData) {
    renderStockData(window.stockLastData, window.stockLastData.ticker);
  }
  if (tab === "portfolio" && window.portfolioLastData) {
    renderPortfolioData(window.portfolioLastData);
  }
  if (tab === "compare" && window.compareLastData) {
    renderComparisonData(window.compareLastData);
  }
};

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    setActiveTab(btn.dataset.tab);
    setGlobalStatus(
      btn.dataset.tab === "stock"
        ? "Stock analysis"
        : btn.dataset.tab === "portfolio"
          ? "Portfolio analysis"
          : "Portfolio comparison",
      "Ready for analysis.",
      false
    );
  });
});

setActiveTab("stock");

const buildRow = (label, value) => {
  const row = document.createElement("div");
  row.innerHTML = `<strong>${label}</strong><span>${value}</span>`;
  row.style.display = "flex";
  row.style.justifyContent = "space-between";
  row.style.gap = "12px";
  return row;
};

const parseHoldings = (input) => {
  if (!input) return [];
  // Split by both comma and newline to be flexible
  return input
    .split(/,|\n/)
    .map((pair) => pair.trim())
    .filter(Boolean)
    .map((pair) => {
      const parts = pair.split(":").map((item) => item.trim());
      if (parts.length < 2) return null;
      const [ticker, weight] = parts;
      return {
        ticker: ticker ? ticker.toUpperCase() : "",
        weight: weight ? parseFloat(weight) : 0,
      };
    })
    .filter((item) => item && item.ticker && !isNaN(item.weight));
};

const getConfig = () => ({
  lookback_days: Number(cfgLookback?.value || 365),
  trading_days: Number(cfgTradingDays?.value || 252),
  risk_free_rate: Number(cfgRiskFree?.value || 0.03),
  risk_free_source: cfgRiskFreeSource?.value || "constant",
  risk_free_ticker: cfgRiskFreeTicker?.value || "^IRX",
  price_interval: cfgInterval?.value || "1d",
  auto_adjust: Boolean(cfgAutoAdjust?.checked),
  short_ma_window: Number(cfgShortMa?.value || 50),
  long_ma_window: Number(cfgLongMa?.value || 200),
  return_method: cfgReturnMethod?.value || "pct",
  drawdown_method: cfgDrawdownMethod?.value || "full",
  drawdown_window_days: Number(cfgDrawdownWindow?.value || 365),
  beta_lookback_days: Number(cfgBetaLookback?.value || 365),
  benchmark_ticker: cfgBenchmark?.value ? cfgBenchmark.value.toUpperCase() : "SPY",
  model: cfgModel?.value || "gpt-4o-mini"
});

const renderPortfolioInsights = (summary) => {
  const wrapper = document.createElement("div");
  if (!summary || summary.status !== "ok") {
    wrapper.textContent =
      summary && summary.message
        ? summary.message
        : "AI summary unavailable.";
    return wrapper;
  }

  const addText = (text, isVerdict = false) => {
    const p = document.createElement("p");
    p.textContent = text;
    p.style.marginBottom = "16px";
    if (isVerdict) {
      p.style.padding = "16px";
      p.style.background = "rgba(154, 208, 255, 0.08)";
      p.style.borderRadius = "12px";
      p.style.borderLeft = "4px solid var(--accent)";
    }
    wrapper.appendChild(p);
  };

  const addList = (title, items) => {
    const h = document.createElement("strong");
    h.textContent = title;
    h.style.display = "block";
    h.style.marginTop = "12px";
    const ul = document.createElement("ul");
    ul.style.margin = "8px 0 16px";
    ul.style.paddingLeft = "20px";
    (items || []).forEach(item => {
      const li = document.createElement("li");
      li.textContent = item;
      ul.appendChild(li);
    });
    wrapper.appendChild(h);
    wrapper.appendChild(ul);
  };

  addText(summary.summary);

  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(300px, 1fr))";
  grid.style.gap = "20px";

  const colA = document.createElement("div");
  const headA = document.createElement("h4");
  headA.textContent = "Current State";
  headA.style.color = "var(--accent)";
  colA.appendChild(headA);
  colA.appendChild(Object.assign(document.createElement("p"), { textContent: summary.current_portfolio_analysis, style: "font-size: 0.9rem; margin-top: 8px;" }));

  const colB = document.createElement("div");
  const headB = document.createElement("h4");
  headB.textContent = "Optimized Alternative";
  headB.style.color = "var(--accent-2)";
  colB.appendChild(headB);
  colB.appendChild(Object.assign(document.createElement("p"), { textContent: summary.suggested_portfolio_analysis, style: "font-size: 0.9rem; margin-top: 8px;" }));

  grid.appendChild(colA);
  grid.appendChild(colB);
  wrapper.appendChild(grid);

  addList("Core Strengths", summary.strengths);
  addList("Risk Factors", summary.weaknesses);
  addList("Strategic Steps", summary.improvements);

  if (summary.comparison) {
    addText(summary.comparison, true);
  }

  return wrapper;
};

const renderHoldingsList = (holdings, minRows = 0) => {
  if ((!holdings || holdings.length === 0) && minRows === 0) return "—";
  const list = document.createElement("div");
  list.style.display = "grid";
  list.style.gap = "10px";

  const items = holdings || [];
  items.forEach((holding) => {
    const row = buildRow(
      `${holding.ticker}${holding.company_name ? ` · ${holding.company_name}` : ""}`,
      `${fmtPct(holding.weight * 100)}`
    );
    list.appendChild(row);
  });

  // Pad to align horizontal lines across columns
  for (let i = items.length; i < minRows; i++) {
    const emptyRow = document.createElement("div");
    emptyRow.innerHTML = "<strong>&nbsp;</strong><span>&nbsp;</span>";
    emptyRow.style.display = "flex";
    emptyRow.style.justifyContent = "space-between";
    emptyRow.style.minHeight = "1.2rem"; // Match standard row height
    list.appendChild(emptyRow);
  }

  return list;
};


const renderInsights = (summary) => {
  const wrapper = document.createElement("div");
  if (!summary || summary.status !== "ok") {
    wrapper.textContent =
      summary && summary.message
        ? summary.message
        : "AI summary unavailable.";
    return wrapper;
  }

  const summaryText = document.createElement("p");
  summaryText.textContent = summary.summary || "No summary returned.";
  summaryText.style.marginBottom = "12px";
  wrapper.appendChild(summaryText);

  const buildList = (title, items) => {
    const label = document.createElement("strong");
    label.textContent = title;
    const list = document.createElement("ul");
    (items || []).forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      list.appendChild(li);
    });
    list.style.margin = "8px 0 16px";
    list.style.paddingLeft = "18px";
    wrapper.appendChild(label);
    wrapper.appendChild(list);
  };

  buildList("Upsides", summary.upsides);
  buildList("Risks", summary.risks);
  buildList("Watch Items", summary.watch_items);

  if (summary.analyst_view) {
    const analyst = document.createElement("p");
    analyst.innerHTML = `<strong>Analyst View</strong><br />${summary.analyst_view}`;
    analyst.style.marginBottom = "12px";
    wrapper.appendChild(analyst);
  }

  return wrapper;
};

const renderNews = (items) => {
  if (!items || items.length === 0) {
    return "No news results yet. Add a BRAVE_API_KEY to enable headlines.";
  }
  const list = document.createElement("div");
  list.style.display = "grid";
  list.style.gap = "16px";
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "news-item card";
    const title = item.title || "Headline";
    const url = item.url || "#";
    const domain = item.domain || "";
    const author = item.author || "";
    const age = item.age || "";
    const description = item.description || "";

    card.innerHTML = `
      <div class="news-meta">
        ${domain ? `<span class="news-domain">${domain}</span>` : ""}
        ${age ? `<span class="news-date">${age}</span>` : ""}
      </div>
      <strong><a href="${url}" target="_blank" rel="noreferrer">${title}</a></strong>
      <p class="news-content">${description}</p>
    `;
    list.appendChild(card);
  });
  return list;
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const ticker = tickerInput.value.trim();
  if (!ticker) return;

  setActiveTab("stock");
  setGlobalStatus("Stock analysis", `Analyzing ${ticker.toUpperCase()}...`, true);
  setStatus("loading", "Analyzing");
  titleEl.textContent = `Analyzing ${ticker.toUpperCase()}`;
  subtitleEl.textContent = "Gathering market data and news...";

  snapshotEl.innerHTML = "";
  momentumEl.innerHTML = "";
  insightsEl.innerHTML = "";
  newsEl.innerHTML = "";

  try {
    const config = getConfig();
    const response = await fetch(
      `/api/analyze?ticker=${ticker}&lookback_days=${config.lookback_days}&trading_days=${config.trading_days}` +
      `&short_ma_window=${config.short_ma_window}&long_ma_window=${config.long_ma_window}` +
      `&price_interval=${config.price_interval}&auto_adjust=${config.auto_adjust}` +
      `&return_method=${config.return_method}&drawdown_method=${config.drawdown_method}` +
      `&drawdown_window_days=${config.drawdown_window_days}&model=${config.model}`
    );
    if (!response.ok) {
      throw new Error("Unable to fetch analysis");
    }
    const data = await response.json();

    renderStockData(data, ticker.toUpperCase());
    window.stockLastData = data;

    setStatus("ready", "Ready");
  } catch (error) {
    setStatus("error", "Error");
    titleEl.textContent = "Analysis failed";
    subtitleEl.textContent = "Please try another ticker.";
    newsEl.textContent = "";
    snapshotEl.replaceChildren();
    momentumEl.replaceChildren();
    insightsEl.replaceChildren();
    setGlobalStatus("Stock analysis", "Analysis failed. Try another ticker.", false);
    return;
  }

  setGlobalStatus("Stock analysis", "Analysis ready.", false);
});

portfolioForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const holdings = parseHoldings(portfolioHoldings.value);
  if (!holdings.length) return;

  setActiveTab("portfolio");
  setGlobalStatus("Portfolio analysis", "Analyzing portfolio...", true);
  setPortfolioStatus("loading", "Analyzing");

  portfolioStatsA.replaceChildren();
  portfolioStatsB.replaceChildren();
  portfolioAiContent.replaceChildren();

  try {
    const config = getConfig();
    const response = await fetch("/api/portfolio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ holdings, ...config }),
    });
    if (!response.ok) {
      throw new Error("Unable to fetch portfolio analysis");
    }
    const data = await response.json();

    renderPortfolioData(data);
    window.portfolioLastData = data;
    setPortfolioStatus("ready", "Ready");
    setGlobalStatus("Portfolio analysis", "Analysis ready.", false);
  } catch (error) {
    setPortfolioStatus("error", "Error");
    setGlobalStatus("Portfolio analysis", "Analysis failed. Check holdings.", false);
  }
});

const renderComparisonData = (data) => {
  const countA = data.portfolio_a.holdings ? data.portfolio_a.holdings.length : 0;
  const countB = data.portfolio_b.holdings ? data.portfolio_b.holdings.length : 0;
  const maxHoldings = Math.max(countA, countB);

  const renderPortfolioOverview = (target, stats, holdings, minRows) => {
    const fragment = document.createDocumentFragment();

    // 1. Holdings first (padded)
    const hList = renderHoldingsList(holdings, minRows);
    if (typeof hList === "string") {
      fragment.append(hList);
    } else {
      fragment.appendChild(hList);
    }

    // 2. Clear Divider
    const hr = document.createElement("hr");
    hr.style.margin = "20px 0";
    hr.style.border = "none";
    hr.style.borderTop = "1px solid rgba(255, 255, 255, 0.12)";
    fragment.appendChild(hr);

    // 3. Stats second
    const statsContainer = document.createElement("div");
    statsContainer.style.display = "grid";
    statsContainer.style.gap = "8px";
    statsContainer.appendChild(buildRow("Annual Return", fmtPct(stats.annualized_return)));
    statsContainer.appendChild(buildRow("Volatility", fmtPct(stats.annualized_volatility)));
    statsContainer.appendChild(buildRow("Max Drawdown", fmtPct(stats.max_drawdown)));
    statsContainer.appendChild(buildRow("Sharpe Ratio", fmtFloat(stats.sharpe_ratio, 2)));
    statsContainer.appendChild(buildRow("Beta", fmtFloat(stats.beta, 2)));
    statsContainer.appendChild(buildRow("Effective Holdings", fmtFloat(stats.effective_holdings, 1)));
    fragment.appendChild(statsContainer);

    target.replaceChildren(fragment);
  };

  renderPortfolioOverview(compareStatsA, data.portfolio_a.stats, data.portfolio_a.holdings, maxHoldings);
  renderPortfolioOverview(compareStatsB, data.portfolio_b.stats, data.portfolio_b.holdings, maxHoldings);

  const wrapper = document.createElement("div");
  const comp = data.ai_comparison;

  if (!comp || comp.status !== "ok") {
    wrapper.textContent = comp?.message || "Comparison analysis unavailable.";
    compareAiContent.replaceChildren(wrapper);
    return;
  }

  const addText = (text, isVerdict = false) => {
    const p = document.createElement("p");
    p.textContent = text;
    p.style.marginBottom = "16px";
    if (isVerdict) {
      p.style.padding = "16px";
      p.style.background = "rgba(255, 211, 139, 0.08)";
      p.style.borderRadius = "12px";
      p.style.borderLeft = "4px solid var(--accent)";
    }
    wrapper.appendChild(p);
  };

  const addList = (title, items) => {
    const h = document.createElement("strong");
    h.textContent = title;
    h.style.display = "block";
    h.style.marginTop = "12px";
    const ul = document.createElement("ul");
    ul.style.margin = "8px 0 16px";
    ul.style.paddingLeft = "20px";
    items.forEach(item => {
      const li = document.createElement("li");
      li.textContent = item;
      ul.appendChild(li);
    });
    wrapper.appendChild(h);
    wrapper.appendChild(ul);
  };

  addText(comp.summary);

  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(300px, 1fr))";
  grid.style.gap = "20px";

  const colA = document.createElement("div");
  const headA = document.createElement("h4");
  headA.textContent = "Portfolio A";
  headA.style.color = "var(--accent)";
  colA.appendChild(headA);
  colA.appendChild(Object.assign(document.createElement("p"), { textContent: comp.portfolio_a_analysis, style: "font-size: 0.9rem; margin-top: 8px;" }));

  const colB = document.createElement("div");
  const headB = document.createElement("h4");
  headB.textContent = "Portfolio B";
  headB.style.color = "var(--accent-2)";
  colB.appendChild(headB);
  colB.appendChild(Object.assign(document.createElement("p"), { textContent: comp.portfolio_b_analysis, style: "font-size: 0.9rem; margin-top: 8px;" }));

  grid.appendChild(colA);
  grid.appendChild(colB);
  wrapper.appendChild(grid);

  const strengthGrid = document.createElement("div");
  strengthGrid.style.display = "grid";
  strengthGrid.style.gridTemplateColumns = "repeat(auto-fit, minmax(300px, 1fr))";
  strengthGrid.style.gap = "20px";
  strengthGrid.style.marginTop = "20px";

  const sA = document.createElement("div");
  const sB = document.createElement("div");

  const buildBulletList = (items) => {
    const ul = document.createElement("ul");
    ul.style.fontSize = "0.85rem";
    ul.style.paddingLeft = "18px";
    ul.style.margin = "8px 0";
    items.forEach(i => {
      const li = document.createElement("li");
      li.textContent = i;
      ul.appendChild(li);
    });
    return ul;
  };

  sA.appendChild(Object.assign(document.createElement("strong"), { textContent: "Strengths A" }));
  sA.appendChild(buildBulletList(comp.strengths_a));
  sA.appendChild(Object.assign(document.createElement("strong"), { textContent: "Weaknesses A" }));
  sA.appendChild(buildBulletList(comp.weaknesses_a));

  sB.appendChild(Object.assign(document.createElement("strong"), { textContent: "Strengths B" }));
  sB.appendChild(buildBulletList(comp.strengths_b));
  sB.appendChild(Object.assign(document.createElement("strong"), { textContent: "Weaknesses B" }));
  sB.appendChild(buildBulletList(comp.weaknesses_b));

  strengthGrid.appendChild(sA);
  strengthGrid.appendChild(sB);
  wrapper.appendChild(strengthGrid);

  addList("Best for Scenarios", comp.best_for_scenarios.map(s => `${s.scenario}: ${s.analysis}`));
  addText(comp.verdict, true);

  compareAiContent.replaceChildren(wrapper);
};

compareForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const holdingsA = parseHoldings(compareAInput.value);
  const holdingsB = parseHoldings(compareBInput.value);

  if (!holdingsA.length || !holdingsB.length) return;

  setActiveTab("compare");
  setGlobalStatus("Portfolio comparison", "Running head-to-head analysis...", true);
  setCompareStatus("loading", "Analyzing");

  compareStatsA.replaceChildren();
  compareStatsB.replaceChildren();
  compareAiContent.replaceChildren();

  try {
    const config = getConfig();
    const response = await fetch("/api/compare-portfolios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        portfolio_a: holdingsA,
        portfolio_b: holdingsB,
        lookback_days: config.lookback_days,
        trading_days: config.trading_days,
        risk_free_rate: config.risk_free_rate,
        price_interval: config.price_interval,
        auto_adjust: config.auto_adjust,
        model: config.model
      }),
    });

    if (!response.ok) {
      throw new Error("Unable to fetch comparison");
    }
    const data = await response.json();
    renderComparisonData(data);
    window.compareLastData = data;
    setCompareStatus("ready", "Ready");
    setGlobalStatus("Portfolio comparison", "Comparison complete.", false);
  } catch (error) {
    setCompareStatus("error", "Error");
    setGlobalStatus("Portfolio comparison", "Comparison failed. Check inputs.", false);
  }
});
