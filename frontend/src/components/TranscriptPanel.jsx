export default function TranscriptPanel({ transcript, keyMoments }) {
  return (
    <section className="rounded-lg border border-borderSoft bg-panel p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">Transcript</h2>
        <span className="rounded-md border border-borderSoft px-2 py-1 text-xs text-[#8888A0]">
          {transcript.length} turns
        </span>
      </div>

      <div className="scrollbar-thin mt-5 max-h-[620px] overflow-y-auto pr-1">
        <div className="flex flex-col gap-3">
          {transcript.map((turn) => {
            const isAgent = turn.speaker.toLowerCase().includes("agent");
            const highlighted = isKeyMoment(turn, keyMoments);
            return (
              <article
                key={turn.id}
                className={`max-w-[92%] rounded-lg border p-3 ${
                  isAgent
                    ? "self-start border-accent/40 bg-accent/10"
                    : "self-end border-borderSoft bg-ink/50"
                } ${highlighted ? "shadow-glow" : ""}`}
              >
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                  <span className={isAgent ? "text-accent" : "text-[#C9C9D6]"}>{turn.speaker}</span>
                  <span className="text-[#8888A0]">{turn.timestamp}</span>
                </div>
                <p className="text-sm leading-6 text-[#EAEAEF]">{turn.text}</p>
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function isKeyMoment(turn, keyMoments) {
  return keyMoments.some((moment) => {
    const description = String(moment.description || "").toLowerCase();
    return description && turn.text.toLowerCase().includes(description.slice(0, 24));
  });
}
