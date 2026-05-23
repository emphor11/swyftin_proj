export function clampScore(score) {
  const numeric = Number(score);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return Math.max(0, Math.min(10, numeric));
}

export function scoreVerdict(score) {
  const value = clampScore(score);
  if (value >= 8.5) return "Excellent";
  if (value >= 7) return "Good";
  if (value >= 5) return "Fair";
  return "Needs work";
}

export function scoreTone(score) {
  const value = clampScore(score);
  if (value >= 7) return "success";
  if (value >= 4) return "warning";
  return "danger";
}

export function titleize(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function parseTranscript(transcriptText = "") {
  return transcriptText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const match = line.match(/^\[([^\]]+)\]\s*([^:]+):\s*(.*)$/);
      if (!match) {
        return {
          id: index,
          timestamp: "-",
          speaker: "Speaker",
          text: line,
        };
      }

      return {
        id: index,
        timestamp: match[1],
        speaker: match[2],
        text: match[3],
      };
    });
}

export function shortDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
