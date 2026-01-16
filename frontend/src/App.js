import axios from "axios";
import React, { useEffect, useState } from "react";
import Select from "react-select";
import "./App.css";

const CPCB = ["pm25", "pm10", "o3", "no2", "so2", "co"];

/* ===========================================================
   CONFIG
   =========================================================== */
const API_BASE_URL = "https://aqi-new-1.onrender.com";

/* ===========================================================
   TAB COMPONENT (Pure React)
   =========================================================== */
function Tabs({ tabs, active, onChange }) {
  return (
    <div className="tabs-container">
      <div className="tab-header">
        {tabs.map((t) => (
          <div
            key={t}
            className={`tab-item ${active === t ? "active" : ""}`}
            onClick={() => onChange(t)}
          >
            {t}
          </div>
        ))}
      </div>
    </div>
  );
}

function App() {
  /* ===========================================================
     BASIC STATE
     =========================================================== */
  const [cities, setCities] = useState([]);
  const [pollutants, setPollutants] = useState([]);
  const [selectedCities, setSelectedCities] = useState([]);
  const [year, setYear] = useState(2025);

  /* ===========================================================
     RESULTS / STATUS
     =========================================================== */
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisDone, setAnalysisDone] = useState(false);
  const [showTests, setShowTests] = useState(false);

  /* ===========================================================
     VISUALIZATION DROPDOWN
     =========================================================== */
  const [visualType, setVisualType] = useState("boxplot");

  const years = Array.from({ length: 11 }, (_, i) => 2015 + i);

  /* ===========================================================
     T-TEST STATE (with Tabs)
     =========================================================== */
  const [tCity1, setTCity1] = useState(null);
  const [tCity2, setTCity2] = useState(null);
  const [tPollutant, setTPollutant] = useState(null);
  const [tResult, setTResult] = useState(null);
  const [tLoading, setTLoading] = useState(false);

  // Tabs for T-Test visuals
  const tTabs = ["Stacked Bar", "Stacked Area", "Density", "Box"];
  const [tActiveTab, setTActiveTab] = useState(tTabs[0]);

  /* ===========================================================
     ANOVA STATE (Continuous + Tukey + Tabs)
     =========================================================== */
  const [aCities, setACities] = useState([]);
  const [aPollutant, setAPollutant] = useState(null);
  const [aResult, setAResult] = useState(null);
  const [aLoading, setALoading] = useState(false);

  const aTabs = ["Box", "Violin", "Tukey Heatmap", "Tukey Table"];
  const [aActiveTab, setAActiveTab] = useState(aTabs[0]);

  /* ===========================================================
     CHI-SQUARE STATE
     =========================================================== */
  const [cCities, setCCities] = useState([]);
  const [cResult, setCResult] = useState(null);
  const [cLoading, setCLoading] = useState(false);

  /* ===========================================================
     INITIAL LOAD — Cities + Pollutants
     =========================================================== */
  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/api/cities`)
      .then((res) =>
        setCities(res.data.cities.map((c) => ({ value: c, label: c })))
      )
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
      .catch(() => {
        setPollutants(["pm25", "pm10", "o3", "no2", "so2", "co"]);
      });
  }, []);

  /* ===========================================================
     ANALYSIS TRIGGER
     =========================================================== */
  const handleAnalyze = async () => {
    if (selectedCities.length === 0) {
      setError("Please select at least one city");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    setAnalysisDone(false);

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

  /* ===========================================================
     TEST PANEL TOGGLE
     =========================================================== */
  const toggleTests = () => setShowTests((prev) => !prev);
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

      {/* -------------------- INPUT PANEL -------------------- */}
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
          }}
          disabled={loading || selectedCities.length === 0}
        >
          {loading ? "⏳ Analyzing..." : "🚀 Run Analysis"}
        </button>
      </div>

      {/* -------------------- SUMMARY SECTIONS -------------------- */}
      {analysisDone && results && (
        <div className="results-container">
          {/* Summary Stats */}
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
          {/* -------------------- VISUALIZATION DROPDOWN -------------------- */}
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
              <option value="boxplot">Box Plot</option>
              <option value="histogram">Histogram</option>
              <option value="density">Density</option>
              <option value="violin">Violin Plot</option>
              <option value="scatter">Scatter (Time)</option>
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
          {/* -------------------- CORRELATION SECTION -------------------- */}
          {results.correlation_matrix && (
            <section className="stats-section">
              <h3>🔗 Correlation Heatmap (Spearman)</h3>
              <img
                src={`data:image/png;base64,${results.visualizations?.correlation_heatmap}`}
                style={{ width: "100%", borderRadius: "10px" }}
                alt="corr-heatmap"
              />

              <h4 style={{ marginTop: "12px" }}>Pairwise Correlations</h4>
              {results.correlation_pairs &&
                Object.entries(results.correlation_pairs).map(
                  ([pair, info], idx) => (
                    <div key={idx} className="corr-row">
                      <strong>{pair.replace("-", " — ")}</strong>: ρ=
                      {info.rho.toFixed(3)}, p={info.p.toFixed(4)}{" "}
                      {info.significant ? (
                        <span className="sig">significant</span>
                      ) : (
                        <span className="nonsig">ns</span>
                      )}
                    </div>
                  )
                )}
            </section>
          )}
          {/* -------------------- NORMALITY + QQ (AQI) -------------------- */}
          {results.qqplots && (
            <section className="viz-section">
              <h3>📉 QQ Plots (AQI + Pollutants)</h3>

              <div className="qq-grid">
                {/* AQI */}
                {results.qqplots.aqi && (
                  <div className="qq-card">
                    <h4>AQI</h4>
                    <img
                      src={`data:image/png;base64,${results.qqplots.aqi}`}
                      style={{ width: "100%", borderRadius: "8px" }}
                    />
                  </div>
                )}

                {/* Pollutants (CPCB Order Guarantee) */}
                {CPCB.filter((p) => results.qqplots[p]).map((p, idx) => (
                  <div key={idx} className="qq-card">
                    <h4>{p.toUpperCase()}</h4>
                    <img
                      src={`data:image/png;base64,${results.qqplots[p]}`}
                      style={{ width: "100%", borderRadius: "8px" }}
                    />
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* =======================================================
    NORMALITY TESTS — GROUPED SECTION
   ======================================================= */}
          <section className="stats-section">
            <h3>🧪 Shapiro Normality Tests</h3>

            {/* AQI */}
            {results.normality_aqi && (
              <div className="normality-row">
                <strong>AQI:</strong> p=
                {results.normality_aqi.pvalue.toFixed(5)} —
                {results.normality_aqi.pvalue < 0.05
                  ? " ❌ Non-normal"
                  : " ✔ Normal"}
              </div>
            )}

            {/* Pollutants */}
            {results.normality_pollutants &&
              Object.entries(results.normality_pollutants).map(
                ([p, v], idx) => (
                  <div key={idx} className="normality-row">
                    <strong>{p.toUpperCase()}:</strong> p={v.pvalue.toFixed(5)}{" "}
                    —{v.is_normal ? " ✔ Normal" : " ❌ Non-normal"}
                  </div>
                )
              )}
          </section>
          {/* =======================================================
               ADVANCED STATISTICAL TESTS PANEL
               ======================================================= */}
          <section className="test-section">
            <h2>🧪 Advanced Statistical Tests</h2>
            <button className="test-btn" onClick={toggleTests}>
              {showTests ? "🔽 Hide Advanced Tests" : "🔼 Show Advanced Tests"}
            </button>

            {showTests && (
              <>
                {/* =======================================================
                     🎯 T-TEST (City vs City) — With TAB VIEW
                     ======================================================= */}
                <section className="test-section">
                  <h3>🎯 Independent T-Test</h3>
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
                    onClick={async () => {
                      setTLoading(true);
                      setTResult(null);
                      try {
                        const payload = {
                          city1: tCity1.value,
                          city2: tCity2.value,
                          pollutant: tPollutant,
                          year,
                        };
                        const res = await axios.post(
                          `${API_BASE_URL}/api/ttest`,
                          payload
                        );
                        setTResult(res.data);
                        setTActiveTab("Stacked Bar");
                      } catch (e) {
                        console.error(e);
                      }
                      setTLoading(false);
                    }}
                  >
                    {tLoading ? "Running..." : "Run T-Test"}
                  </button>

                  {tResult && (
                    <div className="test-result">
                      <p>
                        <strong>Method:</strong> {tResult.method}
                      </p>
                      <p>
                        <strong>T:</strong> {tResult.statistic?.toFixed(4)}
                      </p>
                      <p>
                        <strong>P:</strong> {tResult.pvalue?.toFixed(5)}
                      </p>
                      <p>
                        <strong>Effect Size (d):</strong>{" "}
                        {tResult.effect_size?.toFixed(3)}
                      </p>
                      <p style={{ fontWeight: "600" }}>
                        {tResult.pvalue < 0.05
                          ? "❌ Significant difference (reject H₀)"
                          : "✔ No difference (fail to reject H₀)"}
                      </p>

                      {/* -------- Tabs Navigation -------- */}
                      <Tabs
                        tabs={tTabs}
                        active={tActiveTab}
                        onChange={setTActiveTab}
                      />

                      {/* -------- Tab Content -------- */}
                      {tActiveTab === "Stacked Bar" && tResult.stacked_bar && (
                        <img
                          src={`data:image/png;base64,${tResult.stacked_bar}`}
                          style={{ width: "100%" }}
                        />
                      )}
                      {tActiveTab === "Stacked Area" &&
                        tResult.stacked_area && (
                          <img
                            src={`data:image/png;base64,${tResult.stacked_area}`}
                            style={{ width: "100%" }}
                          />
                        )}
                      {tActiveTab === "Density" && tResult.density_plot && (
                        <img
                          src={`data:image/png;base64,${tResult.density_plot}`}
                          style={{ width: "100%" }}
                        />
                      )}
                      {tActiveTab === "Box" && tResult.box_plot && (
                        <img
                          src={`data:image/png;base64,${tResult.box_plot}`}
                          style={{ width: "100%" }}
                        />
                      )}
                    </div>
                  )}
                </section>

                {/* =======================================================
                     🧮 ANOVA — Continuous F-test + Tukey (Tabs)
                     ======================================================= */}
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
                    onClick={async () => {
                      setALoading(true);
                      setAResult(null);
                      try {
                        const payload = {
                          cities: aCities.map((c) => c.value),
                          pollutant: aPollutant,
                          year,
                        };
                        const res = await axios.post(
                          `${API_BASE_URL}/api/anova`,
                          payload
                        );
                        setAResult(res.data);
                        setAActiveTab("Box");
                      } catch (e) {
                        console.error(e);
                      }
                      setALoading(false);
                    }}
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

                      {/* -------- Tabs Navigation -------- */}
                      <Tabs
                        tabs={aTabs}
                        active={aActiveTab}
                        onChange={setAActiveTab}
                      />

                      {/* -------- Tab Content -------- */}
                      {aActiveTab === "Box" && aResult.box_plot && (
                        <img
                          src={`data:image/png;base64,${aResult.box_plot}`}
                          style={{ width: "100%" }}
                        />
                      )}
                      {aActiveTab === "Violin" && aResult.violin_plot && (
                        <img
                          src={`data:image/png;base64,${aResult.violin_plot}`}
                          style={{ width: "100%" }}
                        />
                      )}
                      {aActiveTab === "Tukey Heatmap" && aResult.tukey_plot && (
                        <img
                          src={`data:image/png;base64,${aResult.tukey_plot}`}
                          style={{ width: "100%" }}
                        />
                      )}
                      {aActiveTab === "Tukey Table" && (
                        <table className="posthoc-table">
                          <thead>
                            <tr>
                              <th>Group 1</th>
                              <th>Group 2</th>
                              <th>Diff</th>
                              <th>P-adj</th>
                              <th>Reject?</th>
                            </tr>
                          </thead>
                          <tbody>
                            {aResult.tukey_table.map((row, idx) => (
                              <tr key={idx}>
                                <td>{row.group1}</td>
                                <td>{row.group2}</td>
                                <td>{row.meandiff.toFixed(2)}</td>
                                <td>{row.p_adj.toFixed(4)}</td>
                                <td>{row.reject ? "Yes" : "No"}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  )}
                </section>
                {/* =======================================================
                     🔢 CHI-SQUARE — City × AQI Category
                     ======================================================= */}
                <section className="test-section">
                  <h3>🔢 Chi-Square Test (City × AQI Category)</h3>

                  <Select
                    isMulti
                    options={cities}
                    onChange={setCCities}
                    placeholder="Select Cities"
                  />

                  <button
                    className="test-btn"
                    disabled={cLoading || cCities.length < 2}
                    onClick={async () => {
                      setCLoading(true);
                      setCResult(null);
                      try {
                        const payload = {
                          cities: cCities.map((c) => c.value),
                          year,
                        };
                        const res = await axios.post(
                          `${API_BASE_URL}/api/chisquare`,
                          payload
                        );
                        setCResult(res.data);
                      } catch (e) {
                        console.error(e);
                      }
                      setCLoading(false);
                    }}
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

                      {cResult.plot && (
                        <img
                          src={`data:image/png;base64,${cResult.plot}`}
                          style={{ width: "100%", borderRadius: "10px" }}
                        />
                      )}
                    </div>
                  )}
                </section>
              </>
            )}
          </section>
          {/* =======================================================
              🤖 AI SUMMARY
              ======================================================= */}
          {results.ai_summary && (
            <section className="summary-section">
              <h3>🤖 AI Summary</h3>
              <pre className="summary-text">{results.ai_summary}</pre>
            </section>
          )}
          {/* =======================================================
              🔮 FORECAST
              ======================================================= */}
          {results.predictions && !results.predictions.error && (
            <section className="predictions-section">
              <h3>🔮 Forecast — {results.predictions.forecast_period} steps</h3>

              <div className="forecast-grid">
                {results.predictions.forecasted_values.map((v, i) => (
                  <div key={i} className="forecast-card">
                    <div className="forecast-value">{v.toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </section>
          )}
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
