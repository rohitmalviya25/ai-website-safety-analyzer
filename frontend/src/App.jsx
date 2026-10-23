
import { useState } from "react";
import {
  ShieldCheck,
  ShieldAlert,
  Search,
  Link2,
  LockKeyhole,
  Activity,
  Clock3,
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
  ArrowUpRight,
  Globe,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api/scan";

const FEATURE_LABELS = {
  url_length: "URL length",
  has_ip_address: "IP address in hostname",
  dot_count: "Number of dots",
  https_flag: "HTTPS detected",
  url_entropy: "URL character entropy",
  token_count: "URL token count",
  subdomain_count: "Subdomain count",
  query_param_count: "Query parameter count",
  tld_length: "TLD length",
  path_length: "Path length",
  has_hyphen_in_domain: "Hyphen in hostname",
  number_of_digits: "Number of digits",
  suspicious_file_extension: "Suspicious file extension",
  domain_name_length: "Hostname length",
  percentage_numeric_chars: "Numeric characters (%)",
};

function getAssessment(score) {
  if (score >= 70) {
    return {
      label: "Higher estimated risk",
      description: "The model identified characteristics associated with phishing URLs.",
      color: "high",
    };
  }

  if (score >= 40) {
    return {
      label: "Moderate estimated risk",
      description: "Review this URL carefully before interacting with it.",
      color: "medium",
    };
  }

  return {
    label: "Lower estimated risk",
    description: "The model estimated a lower phishing risk. This does not guarantee safety.",
    color: "low",
  };
}

function formatFeatureValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }

  return String(value);
}

function App() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastScannedUrl, setLastScannedUrl] = useState("");

  async function handleScan(event) {
    event.preventDefault();

    const submittedUrl = url.trim();

    if (!submittedUrl) {
      setError("Please enter a website URL to analyze.");
      return;
    }

    if (submittedUrl.length > 2048) {
      setError("The URL must be 2048 characters or fewer.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setLastScannedUrl(submittedUrl);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: submittedUrl }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "The scan could not be completed."
        );
      }

      if (
        typeof data.safety_score !== "number" ||
        typeof data.phishing_score !== "number" ||
        !data.features
      ) {
        throw new Error(
          "The API response is missing expected fields. Check your backend response."
        );
      }

      setResult(data);
    } catch (err) {
      if (err instanceof TypeError) {
        setError(
          "Cannot connect to the backend. Make sure FastAPI is running at http://127.0.0.1:8000."
        );
      } else {
        setError(err.message || "Something went wrong.");
      }
    } finally {
      setLoading(false);
    }
  }

  const assessment = result
    ? getAssessment(result.phishing_score)
    : null;

  const safetyScore = result?.safety_score;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" aria-label="SafeWeb home">
          <span className="brand-icon">
            <ShieldCheck size={25} />
          </span>
          <span>
            <strong>SafeWeb</strong>
            <small>WEBSITE INTELLIGENCE</small>
          </span>
        </a>

        <div className="sidebar-label">WORKSPACE</div>

        <a className="nav-item active" href="#scanner">
          <Search size={18} />
          <span>Website Scanner</span>
        </a>

        <a className="nav-item" href="#about">
          <Activity size={18} />
          <span>How it works</span>
        </a>

        <div className="sidebar-bottom">
          <div className="connection-indicator">
            <span className="status-dot" />
            <span>ML scanner interface</span>
          </div>
          <p>AI-powered URL risk assessment</p>
          <small>Academic minor project</small>
        </div>
      </aside>

      <main className="main-content" id="home">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> Website Scanner
          </div>
          <div className="topbar-status">
            <span className="status-dot" />
            <span>URL analysis</span>
          </div>
        </header>

        <section className="page-heading" id="scanner">
          <div>
            <div className="eyebrow">
              <span className="eyebrow-dot" />
              MACHINE LEARNING · CYBERSECURITY
            </div>
            <h1>Website Safety Analyzer</h1>
            <p className="page-description">
              Analyze a URL with a machine learning model and understand
              its estimated phishing risk.
            </p>
          </div>
          <div className="heading-shield">
            <ShieldCheck size={34} />
          </div>
        </section>

        <section className="scanner-card">
          <div className="section-title">
            <span className="section-icon">
              <Link2 size={19} />
            </span>
            <div>
              <h2>Analyze a website</h2>
              <p>Enter a URL to inspect its characteristics.</p>
            </div>
          </div>

          <form onSubmit={handleScan}>
            <label className="input-label" htmlFor="website-url">
              Website URL
            </label>

            <div className="url-input-wrap">
              <Globe size={19} className="url-input-icon" />
              <input
                id="website-url"
                type="text"
                placeholder="https://example.com"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                maxLength={2048}
                autoComplete="url"
                disabled={loading}
              />
              <button
                type="submit"
                className="analyze-button"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <LoaderCircle size={17} className="spin" />
                    Analyzing
                  </>
                ) : (
                  <>
                    <Search size={17} />
                    Analyze website
                  </>
                )}
              </button>
            </div>

            <div className="input-footnote">
              <LockKeyhole size={14} />
              <span>
                The scanner analyzes URL text only. It does not open the
                submitted website.
              </span>
            </div>
          </form>

          {error && (
            <div className="error-message" role="alert">
              <AlertTriangle size={19} />
              <span>{error}</span>
            </div>
          )}
        </section>

        {!result && !loading && !error && (
          <section className="welcome-panel">
            <div className="welcome-icon">
              <Search size={25} />
            </div>
            <h2>Your analysis starts here</h2>
            <p>
              Enter a URL above to see the model's safety estimate,
              phishing-risk score, and extracted features.
            </p>
            <div className="welcome-points">
              <span><CheckCircle2 size={16} /> ML-based prediction</span>
              <span><CheckCircle2 size={16} /> Feature inspection</span>
              <span><CheckCircle2 size={16} /> Instant API response</span>
            </div>
          </section>
        )}

        {loading && (
          <section className="loading-panel" aria-live="polite">
            <LoaderCircle size={32} className="spin" />
            <h2>Analyzing URL characteristics</h2>
            <p>Waiting for the machine learning API to respond...</p>
          </section>
        )}

        {result && assessment && (
          <section className="results-section" aria-live="polite">
            <div className="results-heading">
              <div>
                <div className="eyebrow">SCAN RESULTS</div>
                <h2>Analysis overview</h2>
              </div>
              <span className="result-chip">
                <CheckCircle2 size={15} />
                Analysis complete
              </span>
            </div>

            <div className="results-grid">
              <article className="result-card score-card">
                <div className="card-topline">
                  <span>WEBSITE SAFETY SCORE</span>
                  <ShieldCheck size={19} />
                </div>

                <div className="score-layout">
                  <div
                    className={`score-gauge ${assessment.color}`}
                    style={{
                      "--score-angle": `${safetyScore * 3.6}deg`,
                    }}
                    role="img"
                    aria-label={`Website safety score ${safetyScore} out of 100`}
                  >
                    <div className="gauge-inner">
                      <strong>{safetyScore.toFixed(1)}</strong>
                      <span>OUT OF 100</span>
                    </div>
                  </div>

                  <div className="score-summary">
                    <span className={`risk-badge ${assessment.color}`}>
                      {assessment.label}
                    </span>
                    <p>{assessment.description}</p>
                  </div>
                </div>

                <div className="score-divider" />

                <div className="score-metric">
                  <span>Estimated phishing score</span>
                  <strong>{result.phishing_score.toFixed(1)}/100</strong>
                </div>
                <p className="score-footnote">
                  Both scores are derived from the same model output.
                  They are not calibrated probabilities of real-world safety.
                </p>
              </article>

              <article className="result-card prediction-card">
                <div className="card-topline">
                  <span>MODEL PREDICTION</span>
                  {result.prediction === "likely_phishing" ? (
                    <ShieldAlert size={20} />
                  ) : (
                    <ShieldCheck size={20} />
                  )}
                </div>

                <div
                  className={`prediction-icon ${assessment.color}`}
                >
                  {result.prediction === "likely_phishing" ? (
                    <AlertTriangle size={27} />
                  ) : (
                    <ShieldCheck size={27} />
                  )}
                </div>

                <h3>
                  {result.prediction === "likely_phishing"
                    ? "Likely phishing"
                    : "Likely legitimate"}
                </h3>

                <p>
                  This is the classifier's prediction, not a verification
                  of the website's actual safety.
                </p>

                <div className="prediction-url">
                  <Link2 size={15} />
                  <span>{result.url}</span>
                </div>
              </article>
            </div>
            
            
            {/* VirusTotal reputation results */}
            <article className="result-card features-card virustotal-card">
              <div className="features-heading">
                <div className="section-title">
                  <span className="section-icon">
                    <ShieldCheck size={19} />
                  </span>
                  <div>
                    <h2>VirusTotal Reputation Check</h2>
                    <p>Results from available security-engine reports.</p>
                  </div>
                </div>

                
                
                <span
                  className={`risk-badge ${
                    result.virustotal?.status !== "reported"
                      ? "medium"
                      : (
                          (result.virustotal.malicious ?? 0) > 0 ||
                          (result.virustotal.suspicious ?? 0) > 0
                        )
                        ? "high"
                        : "low"
                  }`}
                >
                  {result.virustotal?.status !== "reported"
                    ? result.virustotal?.status === "not_found"
                      ? "No report found"
                      : "Check unavailable"
                    : (
                        (result.virustotal.malicious ?? 0) > 0 ||
                        (result.virustotal.suspicious ?? 0) > 0
                      )
                      ? "Detections found"
                      : "No detections reported"}
                </span>


              </div>

              {result.virustotal?.status === "reported" ? (
                <>
                  <p className="vt-summary">
                    {result.virustotal.malicious > 0
                      ? `${result.virustotal.malicious} security engine(s) flagged this URL as malicious.`
                      : result.virustotal.suspicious > 0
                        ? `${result.virustotal.suspicious} security engine(s) flagged this URL as suspicious.`
                        : "No malicious or suspicious detections were reported by the engines in this result."}
                  </p>

                  <div className="features-grid">
                    <div className="feature-item">
                      <span>Malicious</span>
                      <strong>{result.virustotal.malicious}</strong>
                    </div>
                    <div className="feature-item">
                      <span>Suspicious</span>
                      <strong>{result.virustotal.suspicious}</strong>
                    </div>
                    <div className="feature-item">
                      <span>Harmless</span>
                      <strong>{result.virustotal.harmless}</strong>
                    </div>
                    <div className="feature-item">
                      <span>Undetected</span>
                      <strong>{result.virustotal.undetected}</strong>
                    </div>
                    <div className="feature-item">
                      <span>Total engine results</span>
                      <strong>{result.virustotal.total_engines}</strong>
                    </div>
                  </div>
                </>
              ) : (
                <p className="vt-summary">
                  {result.virustotal?.message ||
                    "A VirusTotal reputation result is not available for this URL."}
                  {" "}This does not mean the URL is safe.
                </p>
              )}

              <p className="score-footnote">
                VirusTotal results are a separate source of evidence from the ML
                prediction. A clean report does not guarantee that a website is safe.
              </p>
            </article>


            <article className="result-card features-card">
              <div className="features-heading">
                <div className="section-title">
                  <span className="section-icon">
                    <Activity size={19} />
                  </span>
                  <div>
                    <h2>Extracted URL features</h2>
                    <p>
                      Numerical characteristics supplied to the ML model.
                    </p>
                  </div>
                </div>
                <span className="feature-count">
                  {Object.keys(result.features).length} features
                </span>
              </div>

              <div className="features-grid">
                {Object.entries(result.features).map(([key, value]) => (
                  <div className="feature-item" key={key}>
                    <span>
                      {FEATURE_LABELS[key] || key.replaceAll("_", " ")}
                    </span>
                    <strong>{formatFeatureValue(value)}</strong>
                  </div>
                ))}
              </div>
            </article>

            <div className="disclaimer">
              <AlertTriangle size={19} />
              <div>
                <strong>Important safety notice</strong>
                <p>
                  {result.disclaimer} A high score does not prove a URL is
                  malicious, and a low score does not guarantee that it is
                  safe. Do not enter passwords or financial details on a
                  website based solely on this assessment.
                </p>
              </div>
            </div>

            <button
              type="button"
              className="rescan-button"
              onClick={() => {
                setResult(null);
                setError("");
                setUrl("");
                setLastScannedUrl("");
              }}
            >
              <Search size={16} />
              Scan another URL
              <ArrowUpRight size={16} />
            </button>
          </section>
        )}

        <footer className="page-footer" id="about">
          <span>
            <ShieldCheck size={15} /> SafeWeb · AIML Minor Project
          </span>
          <span>
            <Clock3 size={14} /> Model estimates may be incorrect
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;
