// BLOCK 1 START
import axios from "axios";
import React, { useEffect, useState } from "react";
import Select from "react-select";
import "./App.css";

const API_BASE_URL ="https://aqi-new-1.onrender.com";

function App() {
  const [cities, setCities] = useState([]);
  const [pollutants, setPollutants] = useState([]);
  const [selectedCities, setSelectedCities] = useState([]);
  const [year, setYear] = useState(2025);

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [showTests, setShowTests] = useState(false);
  const [analysisDone, setAnalysisDone] = useState(false);

  const [visualType, setVisualType] = useState("boxplot");

  const years = Array.from({ length: 11 }, (_, i) => 2015 + i);

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/cities`)
      .then((res) => {
        setCities(res.data.cities.map((c) => ({ value: c, label: c })));
      })
      .catch(() => {
        setCities(
          [
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Chennai",
            "Kolkata",
            "Hyderabad",
            "Pune",
            "Ahmedabad",
            "Jaipur",
            "Lucknow",
          ].map((c) => ({ value: c, label: c }))
        );
      });
  }, []);

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/pollutants`)
      .then((res) => setPollutants(res.data.pollutants || []))
      .catch((err) => console.error("Pollutant load failed", err));
  }, []);

  const handleAnalyze = async () => {
    if (selectedCities.length === 0) {
      setError("Please select at least one city");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    setAnalysisDone(false);
    setShowTests(false);

    try {
      const payload = {
        cities: selectedCities.map((c) => c.value),
        year: year,
      };
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, payload);
      setResults(response.data);
      setAnalysisDone(true);
    } catch (err) {
      setError(err.response?.data?.detail || "Analysis failed");
    }
    setLoading(false);
  };

  const toggleTests = () => setShowTests((prev) => !prev);

  // --- T-TEST STATE ---
  const [tCity1, setTCity1] = useState(null);
  const [tCity2, setTCity2] = useState(null);
  const [tPollutant, setTPollutant] = useState(null);
  const [tResult, setTResult] = useState(null);
  const [tPlot, setTPlot] = useState(null);
  const [tLoading, setTLoading] = useState(false);

const runTTest = async () => {
  if (!tCity1 || !tCity2 || !tPollutant) return;
  setTLoading(true);
  setTResult(null);
  setTPlot(null);
  try {
    const payload = {
      city1: tCity1.value,
      city2: tCity2.value,
      pollutant: tPollutant,
      year: year
    };
    const res = await axios.post(`${API_BASE_URL}/api/ttest`, payload, { withCredentials: true });
    setTResult(res.data);
    setTPlot(res.data.plot);
  } catch (err) {
    console.error("T-test failed", err);
  }
  setTLoading(false);
};

  // --- ANOVA STATE ---
  const [aCities, setACities] = useState([]);
  const [aPollutant, setAPollutant] = useState(null);
  const [aResult, setAResult] = useState(null);
  const [aPlot, setAPlot] = useState(null);
  const [aPostHoc, setAPostHoc] = useState(null);
  const [aLoading, setALoading] = useState(false);

  const runANOVA = async () => {
  if (aCities.length < 2 || !aPollutant) return;
  setALoading(true);
  setAResult(null);
  setAPlot(null);
  setAPostHoc(null);
  try {
    const payload = {
      cities: aCities.map((c) => c.value),
      pollutant: aPollutant,
      year: year
    };
    const res = await axios.post(`${API_BASE_URL}/api/anova`, payload, { withCredentials: true });
    setAResult(res.data);
    setAPlot(res.data.plot);
    setAPostHoc(res.data.posthoc);
  } catch (err) {
    console.error("ANOVA failed", err);
  }
  setALoading(false);
};

  // --- CHI-SQUARE STATE ---
  const [cCities, setCCities] = useState([]);
  const [cResult, setCResult] = useState(null);
  const [cPlot, setCPlot] = useState(null);
  const [cLoading, setCLoading] = useState(false);

  const runChiSquare = async () => {
  if (cCities.length < 2) return;
  setCLoading(true);
  setCResult(null);
  setCPlot(null);
  try {
    const payload = {
      cities: cCities.map((c) => c.value),
      year: year
    };
    const res = await axios.post(`${API_BASE_URL}/api/chisquare`, payload, { withCredentials: true });
    setCResult(res.data);
    setCPlot(res.data.plot);
  } catch (err) {
    console.error("Chi-square failed", err);
  }
  setCLoading(false);
};

  return (
    <div className="App">
      <header
        style={{
          padding: "30px 0",
          textAlign: "center",
          color: "white",
          fontSize: "42px",
          fontWeight: "700",
        }}
      >
        🌍 India AQI Analysis Dashboard
        <div style={{ fontSize: "18px", opacity: 0.9, marginTop: "8px" }}>
          Real-time Air Quality Index Analysis & Forecasting
        </div>
      </header>

      <div
        style={{
          maxWidth: "900px",
          margin: "20px auto",
          background: "white",
          borderRadius: "18px",
          padding: "35px 40px",
          boxShadow: "0px 8px 25px rgba(0,0,0,0.18)",
          border: "1px solid #eee",
        }}
      >
        <div
          style={{ marginBottom: "25px", fontSize: "18px", fontWeight: 600 }}
        >
          1. Select Cities:
        </div>

        <Select
          options={cities}
          isMulti
          value={selectedCities}
          onChange={setSelectedCities}
          placeholder="Select one or more cities..."
        />

        <div
          style={{
            marginTop: "25px",
            marginBottom: "15px",
            fontSize: "18px",
            fontWeight: 600,
          }}
        >
          2. Select Year:
        </div>

        <select
          value={year}
          onChange={(e) => setYear(Number(e.target.value))}
          style={{
            width: "100%",
            padding: "12px",
            borderRadius: "10px",
            border: "2px solid #ddd",
            fontSize: "16px",
            fontWeight: 500,
          }}
        >
          {years.map((y) => (
            <option key={y} value={y}>
              {y}
            </option>
          ))}
        </select>
        <button
          onClick={handleAnalyze}
          style={{
            width: "100%",
            marginTop: "32px",
            padding: "15px",
            borderRadius: "10px",
            border: "none",
            fontSize: "18px",
            fontWeight: 700,
            cursor: "pointer",
            background: "linear-gradient(to right, #d876f0, #6a62e4)",
            color: "white",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
          }}
        >
          🚀 Run Analysis
        </button>
      </div>

      {/* Reveal statistical sections after analysis */}
      {analysisDone && results && (
        <div className="results-container">
          {/* 📊 Summary Statistics */}
          <section className="stats-section">
            <h3>📊 Summary Statistics</h3>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-label">Mean</span>
                <span className="stat-value">
                  {results.summary_stats.mean?.toFixed(2)}
                </span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Median</span>
                <span className="stat-value">
                  {results.summary_stats.median?.toFixed(2)}
                </span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Std Dev</span>
                <span className="stat-value">
                  {results.variability_metrics.std_dev?.toFixed(2)}
                </span>
              </div>
              <div className="stat-card">
                <span className="stat-label">IQR</span>
                <span className="stat-value">
                  {results.variability_metrics.iqr?.toFixed(2)}
                </span>
              </div>
            </div>
          </section>

          {/* 📦 Visualization Dropdown */}
          <section className="viz-section">
            <h3>📈 Visualizations</h3>
            <select
              value={visualType}
              onChange={(e) => setVisualType(e.target.value)}
              style={{
                padding: "10px",
                borderRadius: "8px",
                border: "2px solid #ddd",
                marginBottom: "10px",
              }}
            >
              <option value="boxplot">Box Plot (Default)</option>
              <option value="histogram">Histogram</option>
              <option value="density">Density</option>
              <option value="violin">Violin Plot</option>
              <option value="scatter">Scatter</option>
              <option value="hexbin">Hexbin</option>
              <option value="correlation_heatmap">Correlation Heatmap</option>
            </select>

            {results.visualizations && results.visualizations[visualType] && (
              <img
                src={`data:image/png;base64,${results.visualizations[visualType]}`}
                style={{ width: "100%", borderRadius: "10px" }}
                alt={visualType}
              />
            )}
          </section>

          {/* 🔗 Correlation Matrix */}
          {results.correlation_matrix?.columns && (
            <section className="stats-section">
              <h3>🔗 Correlation Matrix</h3>
              <div className="correlation-table-container">
                <table className="correlation-table">
                  <thead>
                    <tr>
                      <th></th>
                      {results.correlation_matrix.columns.map((col, i) => (
                        <th key={i}>{col.toUpperCase()}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.correlation_matrix.data.map((row, r) => (
                      <tr key={r}>
                        <th>
                          {results.correlation_matrix.columns[r].toUpperCase()}
                        </th>
                        {row.map((val, c) => (
                          <td key={c}>{val?.toFixed(2)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* 📉 QQ Plot — AQI */}
          {results.qqplot_aqi && (
            <section className="viz-section">
              <h3>📉 QQ Plot — AQI</h3>
              <img
                src={`data:image/png;base64,${results.qqplot_aqi}`}
                style={{ width: "100%", borderRadius: "10px" }}
                alt="qqplot-aqi"
              />
            </section>
          )}

          {/* 🧪 Normality — AQI */}
          {results.normality_aqi && (
            <section className="stats-section">
              <h3>🧪 Normality Test — AQI (Shapiro-Wilk)</h3>
              <p>
                <strong>Statistic:</strong>{" "}
                {results.normality_aqi.statistic?.toFixed(4)}
              </p>
              <p>
                <strong>P-Value:</strong>{" "}
                {results.normality_aqi.pvalue?.toFixed(5)}
              </p>
              <p style={{ marginTop: "6px", fontWeight: "600" }}>
                {results.normality_aqi.pvalue < 0.05
                  ? "❌ AQI is NOT normally distributed (reject H₀)"
                  : "✔ AQI appears normally distributed (fail to reject H₀)"}
              </p>
            </section>
          )}

          {/* 📉 QQ + Normality for Pollutants */}
          {results.qqplots_pollutants &&
            Object.keys(results.qqplots_pollutants).map((p, idx) => (
              <section className="viz-section" key={idx}>
                <h3>📉 QQ Plot — {p.toUpperCase()}</h3>
                <img
                  src={`data:image/png;base64,${results.qqplots_pollutants[p]}`}
                  style={{ width: "100%", borderRadius: "10px" }}
                  alt={`qqplot-${p}`}
                />

                {results.normality_pollutants?.[p] && (
                  <div style={{ marginTop: "10px" }}>
                    <strong>Statistic:</strong>{" "}
                    {results.normality_pollutants[p].statistic?.toFixed(4)}
                    <br />
                    <strong>P-Value:</strong>{" "}
                    {results.normality_pollutants[p].pvalue?.toFixed(5)}
                    <br />
                    <span style={{ marginTop: "5px", fontWeight: "600" }}>
                      {results.normality_pollutants[p].pvalue < 0.05
                        ? "❌ NOT normal (reject H₀)"
                        : "✔ Normal (fail to reject H₀)"}
                    </span>
                  </div>
                )}
              </section>
            ))}

          {/* 🧩 Advanced Tests Toggle */}
          <section className="test-section">
            <h2>🧩 Advanced Statistical Tests</h2>
            <button className="test-btn" onClick={toggleTests}>
              {showTests ? "🔽 Hide Advanced Tests" : "🔼 Show Advanced Tests"}
            </button>

            {showTests && (
              <>
                {/* 🎯 T-TEST SECTION */}
                <section className="test-section">
                  <h3>🎯 Independent T-Test (City vs City)</h3>
                  <div className="test-row">
                    <Select
                      options={cities}
                      placeholder="City 1"
                      onChange={setTCity1}
                    />
                    <Select
                      options={cities}
                      placeholder="City 2"
                      onChange={setTCity2}
                    />
                    <select
                      value={tPollutant || ""}
                      onChange={(e) => setTPollutant(e.target.value)}
                      style={{
                        padding: "10px",
                        borderRadius: "8px",
                        border: "2px solid #ddd",
                      }}
                    >
                      <option value="">Select Pollutant</option>
                      {pollutants.map((p, i) => (
                        <option key={i} value={p}>
                          {p.toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </div>

                  <button
                    className="test-btn"
                    disabled={tLoading || !tCity1 || !tCity2 || !tPollutant}
                    onClick={runTTest}
                  >
                    {tLoading ? "Running..." : "Run T-Test"}
                  </button>

                  {tResult && (
                    <div className="test-result">
                      <p>
                        <strong>T:</strong> {tResult.statistic?.toFixed(4)}
                      </p>
                      <p>
                        <strong>P:</strong> {tResult.pvalue?.toFixed(5)}
                      </p>
                      <p style={{ fontWeight: "600" }}>
                        {tResult.pvalue < 0.05
                          ? "❌ Significant difference (reject H₀)"
                          : "✔ No difference (fail to reject H₀)"}
                      </p>
                    </div>
                  )}

                  {tPlot && (
                    <img src={`data:image/png;base64,${tPlot}`} alt="t-plot" />
                  )}
                </section>

                {/* 🧮 ANOVA */}
                <section className="test-section">
                  <h3>🧮 One-Way ANOVA</h3>
                  <div className="test-row">
                    <Select
                      isMulti
                      options={cities}
                      onChange={setACities}
                      placeholder="Cities"
                    />
                    <select
                      value={aPollutant || ""}
                      onChange={(e) => setAPollutant(e.target.value)}
                      style={{
                        padding: "10px",
                        borderRadius: "8px",
                        border: "2px solid #ddd",
                      }}
                    >
                      <option value="">Select Pollutant</option>
                      {pollutants.map((p, i) => (
                        <option key={i} value={p}>
                          {p.toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </div>

                  <button
                    className="test-btn"
                    disabled={aLoading || aCities.length < 2 || !aPollutant}
                    onClick={runANOVA}
                  >
                    {aLoading ? "Running..." : "Run ANOVA"}
                  </button>

                  {aResult && (
                    <div className="test-result">
                      <p>
                        <strong>F:</strong> {aResult.fstat?.toFixed(4)}
                      </p>
                      <p>
                        <strong>P:</strong> {aResult.pvalue?.toFixed(5)}
                      </p>
                      <p style={{ fontWeight: "600" }}>
                        {aResult.pvalue < 0.05
                          ? "❌ Groups differ (reject H₀)"
                          : "✔ No difference (fail to reject H₀)"}
                      </p>
                    </div>
                  )}

                  {aPlot && (
                    <img
                      src={`data:image/png;base64,${aPlot}`}
                      alt="anova-plot"
                    />
                  )}

                  {aPostHoc && (
                    <>
                      <h4 style={{ marginTop: "10px" }}>
                        Post-Hoc (Tukey HSD)
                      </h4>
                      <img
                        src={`data:image/png;base64,${aPostHoc}`}
                        alt="posthoc"
                      />
                    </>
                  )}
                </section>

                {/* 🔢 CHI-SQUARE */}
                <section className="test-section">
                  <h3>🔢 Chi-Square (City × AQI Category)</h3>
                  <Select
                    isMulti
                    options={cities}
                    onChange={setCCities}
                    placeholder="Select cities"
                  />

                  <button
                    className="test-btn"
                    disabled={cLoading || cCities.length < 2}
                    onClick={runChiSquare}
                  >
                    {cLoading ? "Running..." : "Run Chi-Square"}
                  </button>

                  {cResult && (
                    <div className="test-result">
                      <p>
                        <strong>Chi²:</strong> {cResult.chisq?.toFixed(4)}
                      </p>
                      <p>
                        <strong>P:</strong> {cResult.pvalue?.toFixed(5)}
                      </p>
                      <p style={{ fontWeight: "600" }}>
                        {cResult.pvalue < 0.05
                          ? "❌ Category distributions differ (reject H₀)"
                          : "✔ No difference (fail to reject H₀)"}
                      </p>
                    </div>
                  )}

                  {cPlot && (
                    <img
                      src={`data:image/png;base64,${cPlot}`}
                      alt="chi-plot"
                    />
                  )}
                </section>
              </>
            )}
          </section>

          {/* 🤖 AI Summary */}
          {results.ai_summary && (
            <section className="summary-section">
              <h3>🤖 AI Summary</h3>
              <pre className="summary-text">{results.ai_summary}</pre>
            </section>
          )}

          {/* 🔮 Forecast */}
          {results.predictions && !results.predictions.error && (
            <section className="predictions-section">
              <h3>🔮 Forecast — {results.predictions.forecast_period}</h3>

              {results.predictions.current_aqi && (
                <div className="current-aqi">
                  Current AQI: {results.predictions.current_aqi.toFixed(2)}
                </div>
              )}

              <div className="forecast-grid">
                {results.predictions.forecasted_values.map((val, i) => {
                  const date = results.predictions.dates[i];
                  return (
                    <div key={i} className="forecast-card">
                      <div className="forecast-date">{date}</div>
                      <div className="forecast-value">{val.toFixed(2)}</div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Forecast Error */}
          {results.predictions && results.predictions.error && (
            <section className="predictions-section">
              <h3>Forecast Status</h3>
              <div className="error-message">{results.predictions.error}</div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

export default App;



