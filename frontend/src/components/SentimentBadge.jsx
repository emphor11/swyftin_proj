import { Frown, Meh, Smile } from "lucide-react";

const STYLES = {
  positive: "border-success/40 bg-success/10 text-success",
  negative: "border-danger/40 bg-danger/10 text-danger",
  mixed: "border-warning/40 bg-warning/10 text-warning",
  neutral: "border-borderSoft bg-panelSoft text-[#C9C9D6]",
};

export default function SentimentBadge({ sentiment = "neutral" }) {
  const normalized = String(sentiment || "neutral").toLowerCase();
  const Icon = normalized === "positive" ? Smile : normalized === "negative" ? Frown : Meh;

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium ${STYLES[normalized] || STYLES.neutral}`}>
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {normalized}
    </span>
  );
}
