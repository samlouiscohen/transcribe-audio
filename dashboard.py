#!/usr/bin/env python3
"""
Generate an interactive HTML dashboard to compare transcription passes
side-by-side with an embedded audio player that seeks to each difference.
"""

import argparse
import difflib
import html
import json
import os
import re
import sys
import http.server
import threading
import webbrowser


def parse_transcript(text):
    """Parse a transcript into a list of {timestamp, speaker, text, seconds} entries."""
    entries = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'\[(\d{1,2}):(\d{2})\]\s*(.+?):\s*(.*)', line)
        if m:
            mins, secs, speaker, content = m.groups()
            entries.append({
                "timestamp": f"{int(mins):02d}:{secs}",
                "seconds": int(mins) * 60 + int(secs),
                "speaker": speaker.strip(),
                "text": content.strip(),
                "raw": line,
            })
        else:
            # Continuation or non-standard line
            if entries:
                entries[-1]["text"] += " " + line
                entries[-1]["raw"] += "\n" + line
            else:
                entries.append({
                    "timestamp": "00:00",
                    "seconds": 0,
                    "speaker": "?",
                    "text": line,
                    "raw": line,
                })
    return entries


def align_and_diff(passes):
    """Align transcript passes and identify differences with severity classification."""
    diffs = []
    pass_keys = sorted(passes.keys())

    # Use pass 1 as the reference, compare each line across passes
    ref = passes[pass_keys[0]]
    others = {k: passes[k] for k in pass_keys[1:]}

    # Use difflib to align pass 1 vs each other pass
    for other_key in pass_keys[1:]:
        ref_lines = [e["raw"] for e in ref]
        other_lines = [e["raw"] for e in passes[other_key]]

        sm = difflib.SequenceMatcher(None, ref_lines, other_lines)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue

            ref_block = ref[i1:i2] if i1 < i2 else []
            other_block = passes[other_key][j1:j2] if j1 < j2 else []

            # Classify
            severity = classify_block(ref_block, other_block)

            # Get timestamp from whichever block has content
            ts_seconds = 0
            timestamp = "00:00"
            if ref_block:
                ts_seconds = ref_block[0]["seconds"]
                timestamp = ref_block[0]["timestamp"]
            elif other_block:
                ts_seconds = other_block[0]["seconds"]
                timestamp = other_block[0]["timestamp"]

            diffs.append({
                "severity": severity,
                "timestamp": timestamp,
                "seconds": ts_seconds,
                "pass_a": pass_keys[0],
                "pass_b": other_key,
                "ref_text": "\n".join(e["raw"] for e in ref_block) if ref_block else "(absent)",
                "other_text": "\n".join(e["raw"] for e in other_block) if other_block else "(absent)",
                "ref_speaker": ref_block[0]["speaker"] if ref_block else "",
                "other_speaker": other_block[0]["speaker"] if other_block else "",
            })

    # Deduplicate by timestamp + content (same diff found in multiple pass comparisons)
    seen = set()
    unique_diffs = []
    for d in diffs:
        key = (d["seconds"], d["ref_text"][:80])
        if key not in seen:
            seen.add(key)
            unique_diffs.append(d)

    unique_diffs.sort(key=lambda d: d["seconds"])
    return unique_diffs


def classify_block(ref_block, other_block):
    """Classify a diff block by severity."""
    if ref_block and other_block:
        r_speakers = {e["speaker"] for e in ref_block}
        o_speakers = {e["speaker"] for e in other_block}
        r_text = " ".join(e["text"] for e in ref_block)
        o_text = " ".join(e["text"] for e in other_block)

        # Speaker swap: same-ish content but different speakers
        if r_speakers != o_speakers:
            sim = difflib.SequenceMatcher(None, r_text.lower(), o_text.lower()).ratio()
            if sim > 0.5:
                return "speaker_swap"

        # Timestamp only
        r_no_ts = re.sub(r'\[\d{1,2}:\d{2}\]', '', r_text).strip()
        o_no_ts = re.sub(r'\[\d{1,2}:\d{2}\]', '', o_text).strip()
        if r_no_ts == o_no_ts:
            return "timestamp"

        # Minor vs content
        r_words = re.sub(r'[^\w\s]', '', r_text.lower()).split()
        o_words = re.sub(r'[^\w\s]', '', o_text.lower()).split()
        sim = difflib.SequenceMatcher(None, r_words, o_words).ratio()
        if sim > 0.9:
            return "minor"

    return "content"


SEVERITY_LABELS = {
    "speaker_swap": {"label": "Speaker Misattribution", "color": "#ef4444", "bg": "#fef2f2", "icon": "!!"},
    "content": {"label": "Content Difference", "color": "#f59e0b", "bg": "#fffbeb", "icon": "!"},
    "minor": {"label": "Minor Wording", "color": "#6b7280", "bg": "#f9fafb", "icon": "~"},
    "timestamp": {"label": "Timestamp Only", "color": "#9ca3af", "bg": "#f9fafb", "icon": "·"},
}


def generate_html(diffs, passes, audio_filename):
    """Generate the comparison dashboard HTML."""
    counts = {}
    for d in diffs:
        counts[d["severity"]] = counts.get(d["severity"], 0) + 1

    diffs_json = json.dumps(diffs)
    severity_json = json.dumps(SEVERITY_LABELS)

    pass_keys = sorted(passes.keys())
    pass_word_counts = {k: sum(len(e["text"].split()) for e in passes[k]) for k in pass_keys}

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transcription Comparison Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f0f0f; color: #e5e5e5; }}

  .header {{ position: sticky; top: 0; z-index: 100; background: #171717; border-bottom: 1px solid #2a2a2a; padding: 16px 24px; }}
  .header h1 {{ font-size: 18px; font-weight: 600; color: #fafafa; margin-bottom: 12px; }}

  .audio-section {{ display: flex; align-items: center; gap: 16px; margin-bottom: 12px; }}
  .audio-section audio {{ flex: 1; height: 36px; }}
  .audio-time {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 14px; color: #a3a3a3; min-width: 60px; }}

  .filters {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .filter-btn {{
    padding: 6px 14px; border-radius: 6px; border: 1px solid #333;
    background: #1a1a1a; color: #a3a3a3; cursor: pointer; font-size: 13px;
    transition: all 0.15s; display: flex; align-items: center; gap: 6px;
  }}
  .filter-btn:hover {{ border-color: #555; color: #e5e5e5; }}
  .filter-btn.active {{ border-color: var(--btn-color); color: var(--btn-color); background: var(--btn-bg); }}
  .filter-count {{ font-weight: 700; }}

  .nav-controls {{ display: flex; align-items: center; gap: 8px; margin-left: auto; }}
  .nav-btn {{
    padding: 6px 12px; border-radius: 6px; border: 1px solid #333;
    background: #1a1a1a; color: #a3a3a3; cursor: pointer; font-size: 13px;
  }}
  .nav-btn:hover {{ border-color: #555; color: #e5e5e5; }}
  .nav-info {{ font-size: 13px; color: #737373; min-width: 80px; text-align: center; }}

  .diff-list {{ padding: 16px 24px; display: flex; flex-direction: column; gap: 8px; padding-bottom: 120px; }}

  .diff-card {{
    border: 1px solid #2a2a2a; border-radius: 10px; background: #171717;
    overflow: hidden; cursor: pointer; transition: all 0.15s;
  }}
  .diff-card:hover {{ border-color: #404040; }}
  .diff-card.active {{ border-color: #3b82f6; box-shadow: 0 0 0 1px #3b82f6; }}
  .diff-card.hidden {{ display: none; }}

  .diff-header {{
    display: flex; align-items: center; gap: 12px; padding: 12px 16px;
    border-bottom: 1px solid #222;
  }}
  .diff-severity {{
    padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
  }}
  .diff-timestamp {{
    font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px;
    color: #3b82f6; cursor: pointer;
  }}
  .diff-timestamp:hover {{ text-decoration: underline; }}
  .play-icon {{ font-size: 12px; }}

  .diff-body {{ display: grid; grid-template-columns: 1fr 1fr; }}
  .diff-pass {{
    padding: 14px 16px; font-size: 13px; line-height: 1.6;
    white-space: pre-wrap; word-wrap: break-word;
  }}
  .diff-pass:first-child {{ border-right: 1px solid #222; }}
  .diff-pass-label {{
    font-size: 11px; font-weight: 600; color: #737373;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
  }}
  .diff-pass-text {{ color: #d4d4d4; }}

  .diff-pass.ref .diff-pass-text {{ }}
  .diff-pass.other .diff-pass-text {{ }}

  .word-added {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border-radius: 2px; padding: 0 2px; }}
  .word-removed {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border-radius: 2px; padding: 0 2px; text-decoration: line-through; }}

  .empty-state {{ text-align: center; padding: 60px 24px; color: #525252; }}
  .empty-state h2 {{ font-size: 16px; margin-bottom: 8px; color: #737373; }}

  .stats {{ display: flex; gap: 24px; padding: 0 24px; margin-bottom: 8px; }}
  .stat {{ font-size: 12px; color: #525252; }}
  .stat strong {{ color: #a3a3a3; }}
</style>
</head>
<body>

<div class="header">
  <h1>Transcription Comparison</h1>
  <div class="audio-section">
    <audio id="audio" controls preload="metadata">
      <source src="{audio_filename}" type="audio/mp4">
    </audio>
    <span class="audio-time" id="currentTime">00:00</span>
  </div>
  <div style="display: flex; align-items: center;">
    <div class="filters" id="filters"></div>
    <div class="nav-controls">
      <button class="nav-btn" id="prevBtn" onclick="navDiff(-1)">Prev</button>
      <span class="nav-info" id="navInfo">0 / 0</span>
      <button class="nav-btn" id="nextBtn" onclick="navDiff(1)">Next</button>
    </div>
  </div>
</div>

<div class="stats" id="stats"></div>
<div class="diff-list" id="diffList"></div>

<script>
const DIFFS = {diffs_json};
const SEVERITY = {severity_json};
const PASS_WORDS = {json.dumps(pass_word_counts)};

let activeFilters = new Set(["speaker_swap", "content", "minor", "timestamp"]);
let activeDiffIdx = -1;
let visibleDiffs = [];

const audio = document.getElementById("audio");

// Update time display
audio.addEventListener("timeupdate", () => {{
  const m = Math.floor(audio.currentTime / 60);
  const s = Math.floor(audio.currentTime % 60);
  document.getElementById("currentTime").textContent =
    String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}});

function seekTo(seconds) {{
  audio.currentTime = Math.max(0, seconds - 2); // 2s before for context
  audio.play();
}}

function wordDiff(a, b) {{
  const aWords = a.split(/\\s+/);
  const bWords = b.split(/\\s+/);
  const sm = new difflib.SequenceMatcher(aWords, bWords);
  let aHtml = "", bHtml = "";
  for (const [op, i1, i2, j1, j2] of sm.getOpcodes()) {{
    if (op === "equal") {{
      aHtml += aWords.slice(i1, i2).join(" ") + " ";
      bHtml += bWords.slice(j1, j2).join(" ") + " ";
    }} else if (op === "replace") {{
      aHtml += '<span class="word-removed">' + esc(aWords.slice(i1, i2).join(" ")) + '</span> ';
      bHtml += '<span class="word-added">' + esc(bWords.slice(j1, j2).join(" ")) + '</span> ';
    }} else if (op === "delete") {{
      aHtml += '<span class="word-removed">' + esc(aWords.slice(i1, i2).join(" ")) + '</span> ';
    }} else if (op === "insert") {{
      bHtml += '<span class="word-added">' + esc(bWords.slice(j1, j2).join(" ")) + '</span> ';
    }}
  }}
  return [aHtml, bHtml];
}}

function esc(s) {{ const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }}

// Minimal SequenceMatcher in JS
const difflib = {{
  SequenceMatcher: class {{
    constructor(a, b) {{ this.a = a; this.b = b; }}
    getOpcodes() {{
      const a = this.a, b = this.b;
      const m = a.length, n = b.length;
      // Simple LCS-based diff
      const dp = Array.from({{length: m + 1}}, () => new Array(n + 1).fill(0));
      for (let i = 1; i <= m; i++)
        for (let j = 1; j <= n; j++)
          dp[i][j] = a[i-1] === b[j-1] ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);

      const ops = [];
      let i = m, j = n;
      const raw = [];
      while (i > 0 || j > 0) {{
        if (i > 0 && j > 0 && a[i-1] === b[j-1]) {{
          raw.push(["equal", i-1, i, j-1, j]);
          i--; j--;
        }} else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {{
          raw.push(["insert", i, i, j-1, j]);
          j--;
        }} else {{
          raw.push(["delete", i-1, i, j, j]);
          i--;
        }}
      }}
      raw.reverse();

      // Merge consecutive same-type ops
      const merged = [];
      for (const op of raw) {{
        if (merged.length && merged[merged.length-1][0] === op[0] &&
            merged[merged.length-1][2] === op[1] && merged[merged.length-1][4] === op[3]) {{
          merged[merged.length-1][2] = op[2];
          merged[merged.length-1][4] = op[4];
        }} else {{
          merged.push([...op]);
        }}
      }}

      // Convert delete+insert pairs to replace
      const result = [];
      for (let k = 0; k < merged.length; k++) {{
        if (k + 1 < merged.length && merged[k][0] === "delete" && merged[k+1][0] === "insert") {{
          result.push(["replace", merged[k][1], merged[k][2], merged[k+1][3], merged[k+1][4]]);
          k++;
        }} else {{
          result.push(merged[k]);
        }}
      }}
      return result;
    }}
  }}
}};

function renderFilters() {{
  const el = document.getElementById("filters");
  const counts = {{}};
  DIFFS.forEach(d => counts[d.severity] = (counts[d.severity] || 0) + 1);

  el.innerHTML = Object.entries(SEVERITY).map(([key, s]) => {{
    const count = counts[key] || 0;
    const active = activeFilters.has(key);
    return `<button class="filter-btn ${{active ? 'active' : ''}}"
      style="--btn-color: ${{s.color}}; --btn-bg: ${{s.bg}}20;"
      onclick="toggleFilter('${{key}}')">
      <span class="filter-count">${{count}}</span> ${{s.label}}
    </button>`;
  }}).join("");
}}

function toggleFilter(severity) {{
  if (activeFilters.has(severity)) activeFilters.delete(severity);
  else activeFilters.add(severity);
  renderFilters();
  renderDiffs();
}}

function renderDiffs() {{
  const el = document.getElementById("diffList");
  visibleDiffs = DIFFS.filter(d => activeFilters.has(d.severity));

  if (visibleDiffs.length === 0) {{
    el.innerHTML = '<div class="empty-state"><h2>No differences match filters</h2><p>Adjust filters above</p></div>';
    updateNav();
    return;
  }}

  el.innerHTML = visibleDiffs.map((d, idx) => {{
    const s = SEVERITY[d.severity];
    const [refHtml, otherHtml] = wordDiff(d.ref_text, d.other_text);
    return `<div class="diff-card" id="diff-${{idx}}" onclick="selectDiff(${{idx}})">
      <div class="diff-header">
        <span class="diff-severity" style="background: ${{s.bg}}; color: ${{s.color}};">
          ${{s.icon}} ${{s.label}}
        </span>
        <span class="diff-timestamp" onclick="event.stopPropagation(); seekTo(${{d.seconds}})">
          <span class="play-icon">&#9654;</span> [${{d.timestamp}}]
        </span>
        <span style="font-size: 12px; color: #525252;">Pass ${{d.pass_a}} vs ${{d.pass_b}}</span>
      </div>
      <div class="diff-body">
        <div class="diff-pass ref">
          <div class="diff-pass-label">Pass ${{d.pass_a}} (primary)</div>
          <div class="diff-pass-text">${{refHtml}}</div>
        </div>
        <div class="diff-pass other">
          <div class="diff-pass-label">Pass ${{d.pass_b}}</div>
          <div class="diff-pass-text">${{otherHtml}}</div>
        </div>
      </div>
    </div>`;
  }}).join("");

  updateNav();
}}

function selectDiff(idx) {{
  document.querySelectorAll(".diff-card").forEach(c => c.classList.remove("active"));
  const card = document.getElementById("diff-" + idx);
  if (card) {{
    card.classList.add("active");
    activeDiffIdx = idx;
    updateNav();
  }}
}}

function navDiff(delta) {{
  if (visibleDiffs.length === 0) return;
  let next = activeDiffIdx + delta;
  if (next < 0) next = visibleDiffs.length - 1;
  if (next >= visibleDiffs.length) next = 0;
  selectDiff(next);
  const card = document.getElementById("diff-" + next);
  if (card) card.scrollIntoView({{ behavior: "smooth", block: "center" }});
  seekTo(visibleDiffs[next].seconds);
}}

function updateNav() {{
  document.getElementById("navInfo").textContent =
    visibleDiffs.length === 0 ? "0 / 0" :
    (activeDiffIdx + 1) + " / " + visibleDiffs.length;
}}

// Stats
document.getElementById("stats").innerHTML = Object.entries(PASS_WORDS)
  .map(([k, v]) => `<span class="stat">Pass <strong>${{k}}</strong>: ${{v}} words</span>`).join("");

// Keyboard nav
document.addEventListener("keydown", (e) => {{
  if (e.key === "ArrowDown" || e.key === "j") {{ e.preventDefault(); navDiff(1); }}
  if (e.key === "ArrowUp" || e.key === "k") {{ e.preventDefault(); navDiff(-1); }}
  if (e.key === " " && e.target.tagName !== "BUTTON") {{
    e.preventDefault();
    audio.paused ? audio.play() : audio.pause();
  }}
}});

renderFilters();
renderDiffs();
</script>
</body>
</html>"""


def build_dashboard(pass_files, audio_file, output_html):
    """Build dashboard from pass files and audio."""
    passes = {}
    for i, f in enumerate(pass_files, 1):
        with open(f) as fh:
            passes[i] = parse_transcript(fh.read())

    diffs = align_and_diff(passes)
    audio_basename = os.path.basename(audio_file)

    html_content = generate_html(diffs, passes, audio_basename)
    with open(output_html, "w") as f:
        f.write(html_content)

    return len(diffs)


def serve_and_open(directory, html_file, port=8477):
    """Serve the directory and open the dashboard in a browser."""
    os.chdir(directory)

    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *args: None  # Suppress logs

    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/{os.path.basename(html_file)}"
    print(f"\nDashboard: {url}")
    webbrowser.open(url)

    print("Press Ctrl+C to stop the server.")
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
        print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(description="Visual comparison dashboard for transcription passes")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("pass_files", nargs="+", help="Transcript pass files (2-3 files)")
    parser.add_argument("-o", "--output", help="Output HTML file path")
    parser.add_argument("--no-serve", action="store_true", help="Generate HTML only, don't start server")
    parser.add_argument("-p", "--port", type=int, default=8477, help="Port for local server (default: 8477)")

    args = parser.parse_args()

    if len(args.pass_files) < 2:
        print("Error: need at least 2 pass files to compare", file=sys.stderr)
        sys.exit(1)

    audio_dir = os.path.dirname(os.path.abspath(args.audio_file))
    output_html = args.output or os.path.join(audio_dir, "comparison_dashboard.html")

    # Copy/symlink audio to same dir as HTML if needed
    audio_abs = os.path.abspath(args.audio_file)

    num_diffs = build_dashboard(args.pass_files, audio_abs, output_html)
    print(f"Dashboard generated: {output_html}")
    print(f"Found {num_diffs} differences to review")

    if not args.no_serve:
        serve_and_open(os.path.dirname(output_html), output_html, args.port)


if __name__ == "__main__":
    main()
