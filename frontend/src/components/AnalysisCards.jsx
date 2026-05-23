import { AlertTriangle, CheckCircle2, MoveRight } from "lucide-react";

const SECTIONS = [
  {
    key: "strengths",
    title: "Strengths",
    icon: CheckCircle2,
    tone: "text-success border-success/35 bg-success/10",
  },
  {
    key: "improvement_areas",
    title: "Improvements",
    icon: AlertTriangle,
    tone: "text-warning border-warning/35 bg-warning/10",
  },
  {
    key: "recommended_next_steps",
    title: "Next Steps",
    icon: MoveRight,
    tone: "text-accent border-accent/35 bg-accent/10",
  },
];

export default function AnalysisCards({ analysis }) {
  return (
    <div className="grid gap-5 md:grid-cols-3">
      {SECTIONS.map((section) => (
        <AnalysisSection key={section.key} section={section} items={analysis[section.key] || []} />
      ))}
    </div>
  );
}

function AnalysisSection({ section, items }) {
  const Icon = section.icon;
  return (
    <section className="rounded-lg border border-borderSoft bg-panel p-5">
      <div className="flex items-center gap-2">
        <span className={`grid h-8 w-8 place-items-center rounded-md border ${section.tone}`}>
          <Icon className="h-4 w-4" aria-hidden="true" />
        </span>
        <h2 className="text-base font-semibold">{section.title}</h2>
      </div>
      <div className="mt-5 flex flex-col gap-3">
        {items.length ? (
          items.map((item, index) => (
            <div key={`${section.key}-${index}`} className="rounded-md border border-borderSoft bg-ink/40 p-3 text-sm leading-5 text-[#C9C9D6]">
              {item}
            </div>
          ))
        ) : (
          <p className="text-sm text-[#8888A0]">No items returned.</p>
        )}
      </div>
    </section>
  );
}
