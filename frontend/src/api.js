const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export async function fetchReports() {
  const response = await fetch(`${API_BASE}/api/reports`);
  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, "Failed to load reports"));
  }
  return response.json();
}

export async function fetchReport(reportId) {
  const response = await fetch(`${API_BASE}/api/reports/${reportId}`);
  if (!response.ok) {
    throw new Error(await apiErrorMessage(response, "Failed to load report"));
  }
  return response.json();
}

export function reportDownloadUrl(reportId, format) {
  return `${API_BASE}/api/reports/${reportId}/download/${format}`;
}

export async function analyzeAudio(file, { analyzerMode = "auto", onEvent } = {}) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE}/api/analyze?analyzer_mode=${analyzerMode}`, {
    method: "POST",
    body: form,
  });

  if (!response.ok || !response.body) {
    throw new Error(await apiErrorMessage(response, "Upload failed"));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalEvent = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const messages = buffer.split(/\r?\n\r?\n/);
    buffer = messages.pop() || "";

    for (const message of messages) {
      const event = parseSseMessage(message);
      if (!event) {
        continue;
      }
      onEvent?.(event);
      finalEvent = event;
      if (event.stage === "error") {
        throw new Error(event.message || "Analysis failed");
      }
    }
  }

  return finalEvent;
}

async function apiErrorMessage(response, fallback) {
  const detail = await response.text();
  if (response.status === 500 && !detail.trim()) {
    return `${fallback}: backend API is not running on port 8000`;
  }
  return detail.trim() || `${fallback} (${response.status})`;
}

function parseSseMessage(message) {
  const dataLines = message
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.replace(/^data:\s?/, ""));

  if (!dataLines.length) {
    return null;
  }

  try {
    return JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }
}
