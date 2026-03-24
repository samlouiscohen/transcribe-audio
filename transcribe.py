#!/usr/bin/env python3
"""
Transcribe audio files using Gemini's native audio understanding.
Supports speaker diarization, timestamps, and verbatim transcription.
"""

import argparse
import sys
import os

import google.generativeai as genai


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

    # Speaker labeling
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

FORMAT:
- {speaker_instruction}
- Include a timestamp at the start of each speaker turn, in [MM:SS] format
- Example: [00:00] {example_label}: Hello, how are you?

The transcript should be very long and complete. If it seems short, you are summarizing — go back and include everything."""


def transcribe(audio_path, api_key, model_name, context):
    """Upload audio and transcribe with Gemini."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    print(f"\nUploading {audio_path}...")
    audio_file = genai.upload_file(audio_path)
    print(f"Uploaded: {audio_file.name}")

    prompt = build_prompt(context)

    print("Transcribing (this may take a few minutes)...")
    response = model.generate_content(
        [audio_file, prompt],
        generation_config={"max_output_tokens": 65000},
    )

    return response.text


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

    # Get context
    if args.no_prompt:
        context = {
            "num_speakers": 2,
            "speaker_names": [],
            "description": "",
            "extra": "",
        }
    else:
        context = get_context_from_user()

    # Transcribe
    transcript = transcribe(args.audio_file, args.api_key, args.model, context)

    # Output
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.audio_file)[0]
        output_path = f"{base}_transcript.md"

    with open(output_path, "w") as f:
        f.write(transcript)

    word_count = len(transcript.split())
    print(f"\nDone! {word_count} words")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
