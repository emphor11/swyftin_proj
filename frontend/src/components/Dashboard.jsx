import ArrowLeft from "lucide-react/dist/esm/icons/arrow-left.js";
import Clock from "lucide-react/dist/esm/icons/clock.js";
import ShieldAlert from "lucide-react/dist/esm/icons/shield-alert.js";
import AnalysisCards from "./AnalysisCards.jsx";
import ReportExport from "./ReportExport.jsx";
import ScoreGauge from "./ScoreGauge.jsx";
import SentimentBadge from "./SentimentBadge.jsx";
import TranscriptPanel from "./TranscriptPanel.jsx";
import { parseTranscript, titleize } from "../utils/formatters.js";

export default function Dashboard({ report, onBack }) {
  const analysis = report.analysis || {};
  const transcript = parseTranscript(report.transcript);
  const scoreBreakdown = analysis.score_breakdown || {};
  const keyMoments = analysis.key_moments || [];

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex h-10 w-fit items-center gap-2 rounded-md border border-borderSoft bg-panel px-3 text-sm text-[#C9C9D6] transition hover:bg-panelSoft"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back
        </button>
        <ReportExport reportId={report.id} />
      </div>

      <section className="rounded-lg border border-borderSoft bg-panel p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0 max-w-4xl">
            <div className="flex flex-wrap items-center gap-2">
              <SentimentBadge sentiment={analysis.overall_sentiment} />
              <span className="inline-flex items-center gap-1.5 rounded-md border border-borderSoft px-2.5 py-1 text-xs text-[#8888A0]">
                <Clock className="h-3.5 w-3.5" aria-hidden="true" />
                {analysis.analysis_mode || "llm"}
              </span>
            </div>
            <h2 className="mt-4 text-xl font-semibold tracking-normal">Call Summary</h2>
            <p className="mt-2 text-sm leading-6 text-[#C9C9D6]">{analysis.call_summary || "No summary returned."}</p>
            <p className="mt-3 text-sm text-[#8888A0]">
              Customer journey: {analysis.customer_sentiment_journey || "Not available"}
            </p>
          </div>
          <div className="rounded-lg border border-borderSoft bg-ink/50 p-4">
            <ScoreGauge score={analysis.agent_score} />
          </div>
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-[360px_minmax(0,1fr)]">
        <div className="rounded-lg border border-borderSoft bg-panel p-5">
          <h2 className="text-base font-semibold">Score Breakdown</h2>
          <div className="mt-5 flex flex-col gap-4">
            {Object.entries(scoreBreakdown).map(([key, value]) => (
              <BreakdownBar key={key} label={titleize(key)} value={value} />
            ))}
          </div>
        </div>
        <AnalysisCards analysis={analysis} />
      </section>

      <section className="grid gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(360px,0.85fr)]">
        <TranscriptPanel transcript={transcript} keyMoments={keyMoments} />
        <div className="flex flex-col gap-5">
          <KeyMoments moments={keyMoments} />
          <Compliance flags={analysis.compliance_flags || []} />
        </div>
      </section>
    </div>
  );
}

function BreakdownBar({ label, value }) {
  const score = Math.max(0, Math.min(10, Number(value) || 0));
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-3 text-sm">
        <span className="min-w-0 truncate text-[#C9C9D6]">{label}</span>
        <span className="font-semibold tabular-nums">{score}/10</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-ink">
        <div className="h-full rounded-full bg-accent" style={{ width: `${score * 10}%` }} />
      </div>
    </div>
  );
}

function KeyMoments({ moments }) {
  return (
    <section className="rounded-lg border border-borderSoft bg-panel p-5">
      <h2 className="text-base font-semibold">Key Moments</h2>
      <div className="mt-5 flex flex-col gap-3">
        {moments.length ? (
          moments.map((moment, index) => (
            <div key={`${moment.timestamp_range}-${index}`} className="rounded-lg border border-borderSoft bg-ink/40 p-3">
              <div className="flex flex-wrap items-center gap-2 text-xs text-[#8888A0]">
                <span>{moment.timestamp_range || "-"}</span>
                <span>{moment.speaker || "Speaker"}</span>
                <span className={`rounded-md px-2 py-0.5 ${moment.impact === "negative" ? "bg-danger/15 text-danger" : moment.impact === "positive" ? "bg-success/15 text-success" : "bg-panelSoft text-[#C9C9D6]"}`}>
                  {moment.impact || "neutral"}
                </span>
              </div>
              <p className="mt-2 text-sm leading-5 text-[#C9C9D6]">{moment.description}</p>
            </div>
          ))
        ) : (
          <p className="text-sm text-[#8888A0]">No key moments returned.</p>
        )}
      </div>
    </section>
  );
}

function Compliance({ flags }) {
  const items = flags.length ? flags : ["No compliance issues detected"];
  return (
    <section className="rounded-lg border border-borderSoft bg-panel p-5">
      <div className="flex items-center gap-2">
        <ShieldAlert className="h-4 w-4 text-warning" aria-hidden="true" />
        <h2 className="text-base font-semibold">Compliance Flags</h2>
      </div>
      <div className="mt-4 flex flex-col gap-2">
        {items.map((flag, index) => (
          <div key={`${flag}-${index}`} className="rounded-md border border-borderSoft bg-ink/40 p-3 text-sm text-[#C9C9D6]">
            {flag}
          </div>
        ))}
      </div>
    </section>
  );
}
