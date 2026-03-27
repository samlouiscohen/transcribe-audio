#!/usr/bin/env python3
"""
Transcribe audio files using Gemini's native audio understanding.
Supports speaker diarization, timestamps, verbatim transcription,
and multi-pass temperature-varied verification to surface low-confidence regions.
"""

import argparse
import concurrent.futures
import difflib
import re
import sys
import os

import google.generativeai as genai

DEFAULT_TEMPS = [0.05, 0.2, 0.3]


def get_context_from_user():
    """Prompt the user for context about the audio before transcribing."""
    print("\n--- Audio Context ---")
    print("Provide some context to improve transcription accuracy.\n")

    num_speakers = input("How many speakers? (default: 2): ").strip() or "2"
    speaker_names = input(
        "Speaker names, comma-separated (e.g. 'Sam, Taylor') or leave blank: "
    ).strip()
    who_speaks_first = input(
        "Who speaks first? (name or leave blank if unsure): "
    ).strip()
    description = input(
        "Brief description of the audio (e.g. 'job interview', 'podcast episode'): "
    ).strip()
    extra = input("Any other context or instructions? (optional): ").strip()

    return {
        "num_speakers": int(num_speakers),
        "speaker_names": [s.strip() for s in speaker_names.split(",")]
        if speaker_names
        else [],
        "who_speaks_first": who_speaks_first,
        "description": description,
        "extra": extra,
    }


def build_prompt(context):
    """Build the transcription prompt from user-provided context."""
    num = context["num_speakers"]
    names = context["speaker_names"]
    desc = context["description"]
    extra = context["extra"]
    first = context.get("who_speaks_first", "")

    if names and len(names) == num:
        speaker_instruction = (
            f"There are {num} speakers. Label them as: "
            + ", ".join(f'"{n}"' for n in names)
            + "."
        )
        example_label = names[0]
    else:
        speaker_instruction = (
            f"There are {num} speakers. Label them Speaker 1, Speaker 2, etc."
        )
        example_label = "Speaker 1"

    first_line = f"\nIMPORTANT: {first} is the first person to speak in this recording.\n" if first else ""
    desc_line = f"\nContext: this audio is a {desc}.\n" if desc else ""
    extra_line = f"\nAdditional instructions: {extra}\n" if extra else ""

    return f"""Transcribe this entire audio recording word-for-word, completely verbatim with timestamps.
{first_line}{desc_line}{extra_line}
CRITICAL RULES:
- Do NOT summarize, condense, paraphrase, or skip ANY part of the conversation
- Include every single word spoken, including all filler words (um, uh, like, you know, etc.)
- Include false starts, stutters, repetitions, and incomplete sentences exactly as spoken
- Include every "mm-hmm", "yeah", "right", "okay" — even brief interjections from any speaker
- Do NOT clean up grammar or make the speech sound more polished
- The transcript must cover the ENTIRE recording from start to finish with NOTHING omitted
- If a speaker trails off or restarts a sentence, transcribe that exactly
- Do NOT hallucinate or fabricate any words. If something is unclear, mark it as [inaudible]
- Accuracy is the #1 priority. Only write what was actually said.

NON-VERBAL SOUNDS — capture ALL of these:
- Laughter: [laughs], [short laugh], [extended laughter ~3s], [both laugh], [chuckles]
- Pauses: [pause], [long pause ~5s], [silence ~3s]
- Other sounds: [sighs], [coughs], [clears throat], [paper rustling], [typing sounds], [phone buzzing], [background noise], [door closing]
- Reactions: [exhales], [inhales sharply], [sniffs]
- Include approximate duration for anything lasting more than ~2 seconds, e.g. [laughter ~4s]
- Note if both speakers laugh or react simultaneously, e.g. [both laugh ~3s]

SPEAKER ATTRIBUTION — be extremely careful:
- Pay close attention to which voice is speaking. Each speaker has a distinct voice — do not mix them up.
- If you are unsure who is speaking, mark it as [speaker uncertain]
- When a speaker gives a brief interjection (e.g. "yeah", "mm-hmm") while the other is talking, note it inline: ...sentence [Speaker 2: mm-hmm] continuing sentence...
- Consistency matters: once you identify a voice as a particular speaker, maintain that mapping throughout.

FORMAT:
- {speaker_instruction}
- Include a timestamp at the start of each speaker turn, in [MM:SS] format
- Example: [00:00] {example_label}: Hello, how are you? [short laugh]

The transcript should be very long and complete. If it seems short, you are summarizing — go back and include everything."""


def single_transcribe(audio_file, model_name, prompt, pass_num, temperature, max_retries=2):
    """Run a single transcription pass at the given temperature with retry."""
    for attempt in range(max_retries + 1):
        try:
            label = f"Pass {pass_num}" + (f" (retry {attempt})" if attempt > 0 else "")
            print(f"  {label} started (temp={temperature})...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                [audio_file, prompt],
                generation_config={"max_output_tokens": 65000, "temperature": temperature},
            )
            text = response.text
            word_count = len(text.split())
            print(f"  {label} done ({word_count} words, temp={temperature})")
            return text
        except (ValueError, Exception) as e:
            if attempt < max_retries:
                print(f"  Pass {pass_num} failed ({e}), retrying...")
            else:
                raise


def _extract_speaker(line):
    """Extract the speaker label from a transcript line, if any."""
    m = re.match(r'\[?\d{1,2}:\d{2}\]?\s*(.+?):', line)
    if m:
        return m.group(1).strip()
    return None


def _classify_diff(removed_line, added_line):
    """Classify a diff pair into severity categories."""
    r_speaker = _extract_speaker(removed_line)
    a_speaker = _extract_speaker(added_line)

    # Speaker misattribution — different speaker labels for similar content
    if r_speaker and a_speaker and r_speaker != a_speaker:
        # Strip speaker labels and timestamps to compare content
        r_content = re.sub(r'^\[?\d{1,2}:\d{2}\]?\s*.+?:\s*', '', removed_line)
        a_content = re.sub(r'^\[?\d{1,2}:\d{2}\]?\s*.+?:\s*', '', added_line)
        similarity = difflib.SequenceMatcher(None, r_content, a_content).ratio()
        if similarity > 0.5:
            return "speaker_swap"

    # Timestamp-only difference
    r_no_ts = re.sub(r'\[?\d{1,2}:\d{2}\]?', '', removed_line).strip()
    a_no_ts = re.sub(r'\[?\d{1,2}:\d{2}\]?', '', added_line).strip()
    if r_no_ts == a_no_ts:
        return "timestamp"

    # Minor wording / punctuation
    r_words = re.sub(r'[^\w\s]', '', removed_line.lower()).split()
    a_words = re.sub(r'[^\w\s]', '', added_line.lower()).split()
    similarity = difflib.SequenceMatcher(None, r_words, a_words).ratio()
    if similarity > 0.9:
        return "minor"

    return "content"


def compare_passes(results, temps):
    """Compare all passes and return a categorized diff report."""
    keys = sorted(results.keys())
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]

    report_lines = ["# Transcription Verification Report\n"]
    report_lines.append("## Strategy")
    report_lines.append("Temperature-varied multi-pass: Pass 1 is the primary transcript at low")
    report_lines.append("temperature. Additional passes use higher temperatures to surface")
    report_lines.append("low-confidence regions. Differences are categorized by severity.\n")
    for k in keys:
        report_lines.append(f"- Pass {k}: {len(results[k].split())} words (temp={temps[k - 1]})")
    report_lines.append("")

    total_speaker_swaps = 0
    total_content_diffs = 0
    total_minor_diffs = 0
    total_timestamp_diffs = 0

    for a, b in pairs:
        a_lines = results[a].splitlines()
        b_lines = results[b].splitlines()
        diff = list(difflib.unified_diff(
            a_lines, b_lines, lineterm="",
            fromfile=f"Pass {a} (temp={temps[a - 1]})",
            tofile=f"Pass {b} (temp={temps[b - 1]})"
        ))

        # Pair up removed/added lines for classification
        removed = [l for l in diff if l.startswith("-") and not l.startswith("---")]
        added = [l for l in diff if l.startswith("+") and not l.startswith("+++")]
        speaker_swaps = []
        content_diffs = []
        minor_diffs = []
        timestamp_diffs = []

        # Classify paired diffs
        for r, a_line in zip(removed, added):
            category = _classify_diff(r[1:], a_line[1:])
            pair = (r, a_line)
            if category == "speaker_swap":
                speaker_swaps.append(pair)
            elif category == "content":
                content_diffs.append(pair)
            elif category == "minor":
                minor_diffs.append(pair)
            else:
                timestamp_diffs.append(pair)

        # Handle unpaired lines (insertions/deletions with no match)
        extra = abs(len(removed) - len(added))
        if extra > 0:
            content_diffs.extend(
                [(l, "") for l in (removed if len(removed) > len(added) else added)[min(len(removed), len(added)):]]
            )

        total_speaker_swaps += len(speaker_swaps)
        total_content_diffs += len(content_diffs)
        total_minor_diffs += len(minor_diffs)
        total_timestamp_diffs += len(timestamp_diffs)

        report_lines.append(f"\n## Pass {a} vs Pass {b}\n")

        if speaker_swaps:
            report_lines.append(f"### SPEAKER MISATTRIBUTION ({len(speaker_swaps)} found) — REVIEW THESE\n")
            report_lines.append("The model assigned different speakers to the same content across passes.\n")
            report_lines.append("```diff")
            for r, a_line in speaker_swaps:
                report_lines.append(r)
                report_lines.append(a_line)
            report_lines.append("```\n")

        if content_diffs:
            report_lines.append(f"### Content differences ({len(content_diffs)} found)\n")
            report_lines.append("Different words/phrases — worth spot-checking.\n")
            report_lines.append("```diff")
            for r, a_line in content_diffs:
                report_lines.append(r)
                if a_line:
                    report_lines.append(a_line)
            report_lines.append("```\n")

        if minor_diffs:
            report_lines.append(f"### Minor wording/punctuation ({len(minor_diffs)} found)\n")
            report_lines.append("```diff")
            for r, a_line in minor_diffs:
                report_lines.append(r)
                report_lines.append(a_line)
            report_lines.append("```\n")

        if timestamp_diffs:
            report_lines.append(f"### Timestamp-only ({len(timestamp_diffs)} found) — safe to ignore\n")

        if not any([speaker_swaps, content_diffs, minor_diffs, timestamp_diffs]):
            report_lines.append("No differences — high confidence.\n")

    # Summary at top
    total_all = total_speaker_swaps + total_content_diffs + total_minor_diffs + total_timestamp_diffs
    summary = ["\n## Summary\n"]
    if total_all == 0:
        summary.append("**All passes identical — high confidence across all regions.**\n")
    else:
        if total_speaker_swaps:
            summary.append(f"- **SPEAKER MISATTRIBUTIONS: {total_speaker_swaps}** — review these manually\n")
        if total_content_diffs:
            summary.append(f"- Content differences: {total_content_diffs} — worth spot-checking\n")
        if total_minor_diffs:
            summary.append(f"- Minor wording/punctuation: {total_minor_diffs} — low risk\n")
        if total_timestamp_diffs:
            summary.append(f"- Timestamp-only: {total_timestamp_diffs} — safe to ignore\n")

    report_lines[1:1] = summary

    return "\n".join(report_lines), total_all


def transcribe(audio_path, api_key, model_name, context, passes=3, temps=None):
    """Upload audio and transcribe with Gemini using temperature-varied multi-pass verification."""
    genai.configure(api_key=api_key)

    if temps is None:
        temps = DEFAULT_TEMPS[:passes]
    # Pad with 0.0 if user specified fewer temps than passes
    while len(temps) < passes:
        temps.append(0.0)

    print(f"\nUploading {audio_path}...")
    audio_file = genai.upload_file(audio_path)
    print(f"Uploaded: {audio_file.name}")

    prompt = build_prompt(context)

    if passes == 1:
        text = single_transcribe(audio_file, model_name, prompt, 1, temps[0])
        return text, None, None

    # Multi-pass concurrent transcription with varied temperatures
    print(f"Running {passes} concurrent transcriptions for verification...")
    print(f"  Temperature schedule: {', '.join(str(t) for t in temps[:passes])}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=passes) as executor:
        futures = {
            executor.submit(
                single_transcribe, audio_file, model_name, prompt, i, temps[i - 1]
            ): i
            for i in range(1, passes + 1)
        }
        results = {}
        for future in concurrent.futures.as_completed(futures):
            pass_num = futures[future]
            results[pass_num] = future.result()

    report, total_diffs = compare_passes(results, temps)

    if total_diffs == 0:
        print(f"\nAll {passes} passes are identical — high confidence in accuracy.")
    else:
        print(f"\nFound {total_diffs} low-confidence differences across passes. Review the comparison report.")

    return results[1], results, report


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using Gemini's native audio understanding"
    )
    parser.add_argument("audio_file", help="Path to the audio file (mp3, m4a, wav, etc.)")
    parser.add_argument(
        "-o", "--output", help="Output file path (default: <input>_transcript.md)"
    )
    parser.add_argument(
        "-k", "--api-key", default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "-m", "--model", default="gemini-3.1-pro-preview",
        help="Gemini model to use (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "-p", "--passes", type=int, default=3,
        help="Number of transcription passes (default: 3). Uses varied temperatures to surface low-confidence regions.",
    )
    parser.add_argument(
        "--temps",
        help="Comma-separated temperature schedule (e.g. '0.0,0.2,0.3'). Overrides defaults.",
    )
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="Skip interactive context prompts and use defaults",
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: provide API key via -k flag or GEMINI_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.audio_file):
        print(f"Error: file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    temps = None
    if args.temps:
        temps = [float(t.strip()) for t in args.temps.split(",")]

    if args.no_prompt:
        context = {
            "num_speakers": 2,
            "speaker_names": [],
            "who_speaks_first": "",
            "description": "",
            "extra": "",
        }
    else:
        context = get_context_from_user()

    transcript, all_results, report = transcribe(
        args.audio_file, args.api_key, args.model, context, args.passes, temps
    )

    base = os.path.splitext(args.audio_file)[0]
    output_path = args.output or f"{base}_transcript.md"

    with open(output_path, "w") as f:
        f.write(transcript)

    if all_results:
        for i, text in all_results.items():
            with open(f"{base}_transcript_pass{i}.md", "w") as f:
                f.write(text)
        with open(f"{base}_comparison.md", "w") as f:
            f.write(report)
        print(f"Comparison report: {base}_comparison.md")

    word_count = len(transcript.split())
    print(f"\nDone! {word_count} words")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
