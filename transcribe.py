#!/usr/bin/env python3
"""
Transcribe audio files using Gemini's native audio understanding.
Supports speaker diarization, timestamps, verbatim transcription,
and multi-pass temperature-varied verification to surface low-confidence regions.
"""

import argparse
import concurrent.futures
import difflib
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
    description = input(
        "Brief description of the audio (e.g. 'job interview', 'podcast episode'): "
    ).strip()
    extra = input("Any other context or instructions? (optional): ").strip()

    return {
        "num_speakers": int(num_speakers),
        "speaker_names": [s.strip() for s in speaker_names.split(",")]
        if speaker_names
        else [],
        "description": description,
        "extra": extra,
    }


def build_prompt(context):
    """Build the transcription prompt from user-provided context."""
    num = context["num_speakers"]
    names = context["speaker_names"]
    desc = context["description"]
    extra = context["extra"]

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

    desc_line = f"\nContext: this audio is a {desc}.\n" if desc else ""
    extra_line = f"\nAdditional instructions: {extra}\n" if extra else ""

    return f"""Transcribe this entire audio recording word-for-word, completely verbatim with timestamps.
{desc_line}{extra_line}
CRITICAL RULES:
- Do NOT summarize, condense, paraphrase, or skip ANY part of the conversation
- Include every single word spoken, including all filler words (um, uh, like, you know, etc.)
- Include false starts, stutters, repetitions, and incomplete sentences exactly as spoken
- Include every "mm-hmm", "yeah", "right", "okay" — even brief interjections from any speaker
- Do NOT clean up grammar or make the speech sound more polished
- The transcript must cover the ENTIRE recording from start to finish with NOTHING omitted
- If a speaker trails off or restarts a sentence, transcribe that exactly
- Include all pauses (note them as [pause] where they occur)
- Do NOT hallucinate or fabricate any words. If something is unclear, mark it as [inaudible]
- Accuracy is the #1 priority. Only write what was actually said.

FORMAT:
- {speaker_instruction}
- Include a timestamp at the start of each speaker turn, in [MM:SS] format
- Example: [00:00] {example_label}: Hello, how are you?

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


def compare_passes(results, temps):
    """Compare all passes and return a diff report highlighting low-confidence regions."""
    keys = sorted(results.keys())
    pairs = [(keys[i], keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys))]

    report_lines = ["# Transcription Verification Report\n"]
    report_lines.append("## Strategy")
    report_lines.append("Temperature-varied multi-pass: the primary transcript (Pass 1) is")
    report_lines.append("generated at temp 0.0 for determinism. Additional passes use higher")
    report_lines.append("temperatures to surface low-confidence regions — words/phrases where")
    report_lines.append("the model is uncertain will appear as differences.\n")
    for k in keys:
        report_lines.append(f"- Pass {k}: {len(results[k].split())} words (temp={temps[k - 1]})")
    report_lines.append("")

    # Collect all differing line numbers across pairs
    all_low_confidence = set()
    total_diffs = 0

    for a, b in pairs:
        a_lines = results[a].splitlines()
        b_lines = results[b].splitlines()
        diff = list(difflib.unified_diff(
            a_lines, b_lines, lineterm="",
            fromfile=f"Pass {a} (temp={temps[a - 1]})",
            tofile=f"Pass {b} (temp={temps[b - 1]})"
        ))
        content_diffs = [
            l for l in diff
            if (l.startswith("+") or l.startswith("-"))
            and not l.startswith("+++") and not l.startswith("---")
        ]
        total_diffs += len(content_diffs)

        report_lines.append(f"\n## Pass {a} vs Pass {b} ({len(content_diffs)} differing lines)\n")
        if content_diffs:
            report_lines.append("**[LOW CONFIDENCE]** — These sections varied across temperature passes:\n")
            report_lines.append("```diff")
            for line in diff:
                if line.startswith("@@") or line.startswith("+") or line.startswith("-"):
                    report_lines.append(line)
            report_lines.append("```")
        else:
            report_lines.append("No differences — high confidence in this comparison.")

    if total_diffs == 0:
        report_lines.insert(2, "\n**RESULT: All passes identical — high confidence across all regions.**\n")
    else:
        report_lines.insert(2, f"\n**RESULT: {total_diffs} differences found — review [LOW CONFIDENCE] sections below.**\n")

    return "\n".join(report_lines), total_diffs


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
