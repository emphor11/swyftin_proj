import { useRef, useState } from "react";
import FileAudio from "lucide-react/dist/esm/icons/file-audio.js";
import FolderOpen from "lucide-react/dist/esm/icons/folder-open.js";
import History from "lucide-react/dist/esm/icons/history.js";
import UploadCloud from "lucide-react/dist/esm/icons/upload-cloud.js";
import { shortDate } from "../utils/formatters.js";

export default function UploadScreen({ latestReport, reports, onUpload, onLoadReport }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  function handleFiles(files) {
    const file = files?.[0];
    if (file) {
      onUpload(file);
    }
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
      <section
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          handleFiles(event.dataTransfer.files);
        }}
        className={`relative min-h-[520px] overflow-hidden rounded-lg border border-dashed p-5 transition ${
          isDragging ? "border-accent bg-accent/10" : "border-borderSoft bg-panel"
        }`}
      >
        <WaveField />
        <div className="relative z-10 flex h-full min-h-[480px] flex-col items-center justify-center gap-5 text-center">
          <div className="grid h-20 w-20 place-items-center rounded-lg border border-borderSoft bg-ink/80">
            <UploadCloud className="h-9 w-9 text-accent" aria-hidden="true" />
          </div>
          <div className="max-w-xl">
            <h2 className="text-2xl font-semibold tracking-normal sm:text-3xl">
              Drop your call recording here
            </h2>
            <p className="mt-3 text-sm leading-6 text-[#8888A0]">
              MP3, WAV, M4A, FLAC, OGG, and WEBM files are accepted.
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2">
            {[".mp3", ".wav", ".m4a", ".flac"].map((format) => (
              <span key={format} className="rounded-md border border-borderSoft bg-panelSoft px-2.5 py-1 text-xs text-[#C9C9D6]">
                {format}
              </span>
            ))}
          </div>

          <input
            ref={inputRef}
            type="file"
            accept=".mp3,.wav,.m4a,.flac,.ogg,.webm,audio/*"
            className="hidden"
            onChange={(event) => handleFiles(event.target.files)}
          />
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="inline-flex h-11 items-center gap-2 rounded-md bg-accent px-4 text-sm font-semibold text-white transition hover:bg-[#5F50D2]"
          >
            <FolderOpen className="h-4 w-4" aria-hidden="true" />
            Choose File
          </button>
        </div>
      </section>

      <aside className="flex flex-col gap-5">
        <section className="rounded-lg border border-borderSoft bg-panel p-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <History className="h-4 w-4 text-accent" aria-hidden="true" />
              <h2 className="text-sm font-semibold">Recent Reports</h2>
            </div>
            <span className="text-xs text-[#8888A0]">{reports.length}</span>
          </div>

          <div className="mt-4 flex flex-col gap-2">
            {reports.slice(0, 6).map((report) => (
              <button
                key={report.id}
                type="button"
                onClick={() => onLoadReport(report.id)}
                className="group grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 rounded-md border border-borderSoft bg-ink/40 p-3 text-left transition hover:border-accent/60 hover:bg-panelSoft"
              >
                <FileAudio className="h-4 w-4 text-[#8888A0] group-hover:text-accent" aria-hidden="true" />
                <span className="min-w-0">
                  <span className="block truncate text-sm text-[#EAEAEF]">{report.id}</span>
                  <span className="mt-0.5 block truncate text-xs text-[#8888A0]">{shortDate(report.created_at)}</span>
                </span>
                <span className="rounded-md border border-borderSoft px-2 py-1 text-xs text-[#C9C9D6]">
                  {report.agent_score ?? "-"}
                </span>
              </button>
            ))}
            {!reports.length ? (
              <div className="rounded-md border border-borderSoft bg-ink/40 p-4 text-sm text-[#8888A0]">
                No saved reports yet.
              </div>
            ) : null}
          </div>
        </section>

        {latestReport ? (
          <button
            type="button"
            onClick={() => onLoadReport(latestReport.id)}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-md border border-borderSoft bg-panel px-4 text-sm font-semibold text-[#EAEAEF] transition hover:bg-panelSoft"
          >
            <FileAudio className="h-4 w-4 text-accent" aria-hidden="true" />
            Open Latest Report
          </button>
        ) : null}
      </aside>
    </div>
  );
}

function WaveField() {
  const bars = Array.from({ length: 34 }, (_, index) => index);
  return (
    <div className="absolute inset-x-6 bottom-8 top-8 opacity-35" aria-hidden="true">
      <div className="flex h-full items-center justify-center gap-1.5">
        {bars.map((bar) => (
          <span
            key={bar}
            className="w-1 rounded-full bg-accent/70"
            style={{
              height: `${18 + ((bar * 17) % 70)}%`,
              opacity: 0.25 + ((bar * 13) % 60) / 100,
            }}
          />
        ))}
      </div>
    </div>
  );
}
