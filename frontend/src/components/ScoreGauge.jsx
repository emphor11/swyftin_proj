import { clampScore, scoreTone, scoreVerdict } from "../utils/formatters.js";

const TONE_COLORS = {
  success: "#00D68F",
  warning: "#FFB800",
  danger: "#FF6B6B",
};

export default function ScoreGauge({ score }) {
  const value = clampScore(score);
  const radius = 56;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 10) * circumference;
  const color = TONE_COLORS[scoreTone(value)];

  return (
    <div className="relative grid h-56 w-56 place-items-center">
      <svg viewBox="0 0 140 140" className="h-52 w-52 -rotate-90" role="img" aria-label={`Agent score ${value} out of 10`}>
        <circle cx="70" cy="70" r={radius} fill="none" stroke="#242430" strokeWidth="12" />
        <circle
          cx="70"
          cy="70"
          r={radius}
          fill="none"
          stroke={color}
          strokeLinecap="round"
          strokeWidth="12"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="absolute text-center">
        <div className="text-5xl font-semibold tabular-nums">{value}</div>
        <div className="mt-1 text-sm text-[#8888A0]">{scoreVerdict(value)}</div>
      </div>
    </div>
  );
}
