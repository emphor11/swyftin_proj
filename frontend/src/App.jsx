import { useEffect, useMemo, useState } from "react";
import Activity from "lucide-react/dist/esm/icons/activity.js";
import AlertCircle from "lucide-react/dist/esm/icons/alert-circle.js";
import BarChart3 from "lucide-react/dist/esm/icons/bar-chart-3.js";
import Loader2 from "lucide-react/dist/esm/icons/loader-2.js";
import RefreshCw from "lucide-react/dist/esm/icons/refresh-cw.js";
import { analyzeAudio, fetchReport, fetchReports } from "./api.js";
import UploadScreen from "./components/UploadScreen.jsx";
import PipelineProgress from "./components/PipelineProgress.jsx";
import Dashboard from "./components/Dashboard.jsx";

export default function App() {
  const [reports, setReports] = useState([]);
  const [currentReport, setCurrentReport] = useState(null);
  const [progressEvents, setProgressEvents] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");
  const [analyzerMode, setAnalyzerMode] = useState("heuristic");
  const [isLoadingReports, setIsLoadingReports] = useState(true);

  useEffect(() => {
    refreshReports();
  }, []);

  const latestReport = useMemo(() => reports[0] || null, [reports]);

  async function refreshReports() {
    setIsLoadingReports(true);
    try {
      const data = await fetchReports();
      setReports(data.reports || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoadingReports(false);
    }
  }

  async function loadReport(reportId) {
    setError("");
    try {
      const report = await fetchReport(reportId);
      setCurrentReport(report);
      setIsProcessing(false);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleUpload(file) {
    setError("");
    setCurrentReport(null);
    setProgressEvents([]);
    setIsProcessing(true);

    try {
      const finalEvent = await analyzeAudio(file, {
        analyzerMode,
        onEvent: (event) => setProgressEvents((events) => [...events, event]),
      });

      if (finalEvent?.report_id) {
        await loadReport(finalEvent.report_id);
        await refreshReports();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  }

  return (
    <main className="min-h-screen px-4 py-5 text-[#EAEAEF] sm:px-6 lg:px-8">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-5">
        <header className="flex flex-col gap-4 border-b border-borderSoft pb-5 md:flex-row md:items-center md:justify-between">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-borderSoft bg-panel shadow-glow">
              <BarChart3 className="h-5 w-5 text-accent" aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-xl font-semibold tracking-normal sm:text-2xl">
                Voice Call Analysis
              </h1>
              <p className="mt-1 text-sm text-[#8888A0]">
                Coaching report workspace
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <ModeToggle value={analyzerMode} onChange={setAnalyzerMode} />
            <button
              type="button"
              onClick={refreshReports}
              className="inline-flex h-10 items-center gap-2 rounded-md border border-borderSoft bg-panel px-3 text-sm text-[#C9C9D6] transition hover:bg-panelSoft"
            >
              {isLoadingReports ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <RefreshCw className="h-4 w-4" aria-hidden="true" />
              )}
              Refresh
            </button>
          </div>
        </header>

        {error ? (
          <div className="flex items-start gap-3 rounded-lg border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-[#FFD6D6]">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <span>{error}</span>
          </div>
        ) : null}

        {isProcessing ? (
          <PipelineProgress events={progressEvents} />
        ) : currentReport ? (
          <Dashboard report={currentReport} onBack={() => setCurrentReport(null)} />
        ) : (
          <UploadScreen
            latestReport={latestReport}
            reports={reports}
            onUpload={handleUpload}
            onLoadReport={loadReport}
          />
        )}
      </div>
    </main>
  );
}

function ModeToggle({ value, onChange }) {
  const options = [
    { value: "auto", label: "Auto", icon: Activity },
    { value: "heuristic", label: "Fast", icon: Loader2 },
    { value: "llm", label: "Phi-3", icon: BarChart3 },
  ];

  return (
    <div className="grid h-10 grid-cols-3 rounded-lg border border-borderSoft bg-panel p-1">
      {options.map((option) => {
        const Icon = option.icon;
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={`inline-flex min-w-16 items-center justify-center gap-1.5 rounded-md px-2 text-xs font-medium transition ${
              active ? "bg-accent text-white" : "text-[#8888A0] hover:bg-panelSoft hover:text-[#EAEAEF]"
            }`}
          >
            <Icon className={`h-3.5 w-3.5 ${option.value === "heuristic" ? "" : ""}`} aria-hidden="true" />
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
