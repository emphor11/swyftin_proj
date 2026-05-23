/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0A0A0F",
        panel: "#13131A",
        panelSoft: "#1A1A24",
        borderSoft: "#2A2A35",
        accent: "#6C5CE7",
        success: "#00D68F",
        warning: "#FFB800",
        danger: "#FF6B6B",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(108, 92, 231, 0.25), 0 24px 80px rgba(0, 0, 0, 0.35)",
      },
    },
  },
  plugins: [],
};
