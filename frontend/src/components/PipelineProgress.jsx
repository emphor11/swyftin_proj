import Check from "lucide-react/dist/esm/icons/check.js";
import Loader2 from "lucide-react/dist/esm/icons/loader-2.js";
import RadioTower from "lucide-react/dist/esm/icons/radio-tower.js";

const STAGES = [
  { key: "normalizing", label: "Normalizing Audio", threshold: 10 },
  { key: "transcribing", label: "Transcribing Speech", threshold: 30 },
  { key: "diarizing", label: "Identifying Speakers", threshold: 50 },
  { key: "merging", label: "Merging Transcript", threshold: 65 },
  { key: "analyzing", label: "Analyzing with AI", threshold: 82 },
  { key: "generating_report", label: "Generating Report", threshold: 95 },
];

export default function PipelineProgress({ events }) {
  const latest = events[events.length - 1] || { stage: "queued", progress: 0, message: "Queued" };
  const progress = Number(latest.progress || 0);

  return (
    <section className="rounded-lg border border-borderSoft bg-panel p-5 shadow-glow">
      <div className="flex flex-col gap-3 border-b border-borderSoft pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-lg bg-accent/15">
            <RadioTower className="h-5 w-5 text-accent" aria-hidden="true" />
          </div>
          <div>
            <h2 className="text-lg font-semibold">Pipeline Progress</h2>
            <p className="mt-1 text-sm text-[#8888A0]">{latest.message || latest.stage}</p>
          </div>
        </div>
        <div className="text-3xl font-semibold tabular-nums">{progress}%</div>
      </div>

      <div className="mt-5 h-2 overflow-hidden rounded-full bg-ink">
        <div className="h-full rounded-full bg-accent transition-all duration-500" style={{ width: `${progress}%` }} />
      </div>

      <div className="mt-6 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {STAGES.map((stage) => {
          const isComplete = progress > stage.threshold;
          const isActive = latest.stage === stage.key;
          return (
            <div
              key={stage.key}
              className={`flex min-h-20 items-center gap-3 rounded-lg border p-4 transition ${
                isActive
                  ? "border-accent bg-accent/10"
                  : isComplete
                    ? "border-success/40 bg-success/10"
                    : "border-borderSoft bg-ink/40"
              }`}
            >
              <div
                className={`grid h-9 w-9 shrink-0 place-items-center rounded-md ${
                  isComplete ? "bg-success text-ink" : isActive ? "bg-accent text-white" : "bg-panelSoft text-[#8888A0]"
                }`}
              >
                {isComplete ? (
                  <Check className="h-4 w-4" aria-hidden="true" />
                ) : isActive ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <span className="h-2 w-2 rounded-full bg-current" />
                )}
              </div>
              <div className="min-w-0">
                <h3 className="truncate text-sm font-semibold">{stage.label}</h3>
                <p className="mt-1 text-xs text-[#8888A0]">{isActive ? "Running" : isComplete ? "Complete" : "Pending"}</p>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
