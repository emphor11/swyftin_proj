import Download from "lucide-react/dist/esm/icons/download.js";
import { reportDownloadUrl } from "../api.js";

export default function ReportExport({ reportId }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <ExportButton href={reportDownloadUrl(reportId, "json")} label="JSON" />
      <ExportButton href={reportDownloadUrl(reportId, "md")} label="Markdown" />
      <ExportButton href={reportDownloadUrl(reportId, "transcript")} label="Transcript" />
    </div>
  );
}

function ExportButton({ href, label }) {
  return (
    <a
      href={href}
      className="inline-flex h-10 items-center gap-2 rounded-md border border-borderSoft bg-panel px-3 text-sm text-[#C9C9D6] transition hover:bg-panelSoft"
    >
      <Download className="h-4 w-4 text-accent" aria-hidden="true" />
      {label}
    </a>
  );
}
