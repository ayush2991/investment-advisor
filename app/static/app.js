if (window.__atlasInitialized) {
  console.warn("Atlas UI already initialized.");
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
const portfolioStats = document.getElementById("portfolio-stats");
const portfolioInsights = document.getElementById("portfolio-insights");
const portfolioSuggestedStats = document.getElementById("portfolio-suggested-stats");
const portfolioHoldingsList = document.getElementById("portfolio-holdings-list");
const portfolioSuggestedHoldings = document.getElementById("portfolio-suggested-holdings");
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
  const stats = document.createDocumentFragment();
  stats.appendChild(buildRow("Annualized Return", fmtPct(data.stats.annualized_return)));
  stats.appendChild(buildRow("Annualized Volatility", fmtPct(data.stats.annualized_volatility)));
  stats.appendChild(buildRow("Max Drawdown", fmtPct(data.stats.max_drawdown)));
  stats.appendChild(buildRow("Beta", fmtFloat(data.stats.beta)));
  stats.appendChild(buildRow("Concentration (HHI)", fmtFloat(data.stats.concentration_hhi, 3)));
  stats.appendChild(buildRow("Effective Holdings", fmtFloat(data.stats.effective_holdings, 2)));
  stats.appendChild(buildRow("Sharpe Ratio", fmtFloat(data.stats.sharpe_ratio, 2)));
  portfolioStats.replaceChildren(stats);

  portfolioInsights.replaceChildren(renderPortfolioInsights(data.ai_summary));

  if (data.suggested_portfolio && data.suggested_portfolio.stats && portfolioSuggestedStats) {
    const suggestedStats = document.createDocumentFragment();
    const s = data.suggested_portfolio.stats;
    suggestedStats.appendChild(buildRow("Annualized Return", fmtPct(s.annualized_return)));
    suggestedStats.appendChild(buildRow("Annualized Volatility", fmtPct(s.annualized_volatility)));
    suggestedStats.appendChild(buildRow("Max Drawdown", fmtPct(s.max_drawdown)));
    suggestedStats.appendChild(buildRow("Beta", fmtFloat(s.beta)));
    suggestedStats.appendChild(buildRow("Concentration (HHI)", fmtFloat(s.concentration_hhi, 3)));
    suggestedStats.appendChild(buildRow("Effective Holdings", fmtFloat(s.effective_holdings, 2)));
    suggestedStats.appendChild(buildRow("Sharpe Ratio", fmtFloat(s.sharpe_ratio, 2)));
    portfolioSuggestedStats.replaceChildren(suggestedStats);
  }

  if (portfolioHoldingsList) {
    portfolioHoldingsList.replaceChildren(renderHoldingsList(data.holdings));
  }
  if (portfolioSuggestedHoldings && data.suggested_holdings) {
    const enriched = data.suggested_holdings.map((holding) => ({
      ticker: holding.ticker,
      weight: holding.weight,
    }));
    portfolioSuggestedHoldings.replaceChildren(renderHoldingsList(enriched));
  }
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
};

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    setActiveTab(btn.dataset.tab);
    setGlobalStatus(
      btn.dataset.tab === "stock" ? "Stock analysis" : "Portfolio analysis",
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
  return input
    .split(",")
    .map((pair) => pair.trim())
    .filter(Boolean)
    .map((pair) => {
      const [ticker, weight] = pair.split(":").map((item) => item.trim());
      return {
        ticker: ticker ? ticker.toUpperCase() : "",
        weight: weight ? Number(weight) : 0,
      };
    })
    .filter((item) => item.ticker);
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

  buildList("Strengths", summary.strengths);
  buildList("Weaknesses", summary.weaknesses);
  buildList("Improvements", summary.improvements);

  if (summary.comparison) {
    const comparison = document.createElement("p");
    comparison.innerHTML = `<strong>Comparison</strong><br />${summary.comparison}`;
    wrapper.appendChild(comparison);
  }

  return wrapper;
};

const renderHoldingsList = (holdings) => {
  if (!holdings || holdings.length === 0) return "—";
  const list = document.createElement("div");
  list.style.display = "grid";
  list.style.gap = "10px";
  holdings.forEach((holding) => {
    const row = buildRow(
      `${holding.ticker}${holding.company_name ? ` · ${holding.company_name}` : ""}`,
      `${fmtPct(holding.weight * 100)}`
    );
    list.appendChild(row);
  });
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
      `&drawdown_window_days=${config.drawdown_window_days}`
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

  portfolioStats.replaceChildren();
  portfolioInsights.replaceChildren();
  if (portfolioSuggestedStats) portfolioSuggestedStats.replaceChildren();
  if (portfolioHoldingsList) portfolioHoldingsList.replaceChildren();
  if (portfolioSuggestedHoldings) portfolioSuggestedHoldings.replaceChildren();

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
