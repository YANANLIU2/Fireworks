#!/usr/bin/env python3
"""
Batch Fireworks Rhythm Encoder
------------------------------
Processes all audio files under 'music_source/' and saves encoded results
into 'runs/' with the same base filename.

Usage:
    python fireworks_batch_encoder.py
"""

import os
import librosa
import numpy as np
from typing import List, Dict, Tuple

# Symbol set (weak -> strong)
DEFAULT_SYMBOLS = list(".:-=*#")

# --- helpers ---
def mmssms(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{m:02d}:{s:02d}.{ms:03d}"

def robust_minmax(x: np.ndarray, lo_p=5.0, hi_p=95.0) -> Tuple[float, float]:
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    if hi <= lo:
        hi = lo + 1e-9
    return lo, hi

def quantize_levels(values: np.ndarray, n_levels: int) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    levels = np.floor(clipped * n_levels).astype(int)
    return np.minimum(levels, n_levels - 1)

def onset_events(y: np.ndarray, sr: int, hop_length: int = 512):
    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Detect onset frames (built-in peak picking, API safe)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames"
    )

    # Convert frames -> times
    times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Get corresponding strengths
    strengths = onset_env[onset_frames]

    return times, strengths


def encode(audio_path: str, symbols: List[str] = DEFAULT_SYMBOLS) -> List[Dict]:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    times, strengths = onset_events(y, sr)
    if len(times) == 0:
        return []

    lo, hi = robust_minmax(strengths)
    norm = (strengths - lo) / (hi - lo)
    levels = quantize_levels(norm, len(symbols))

    return [
        {"time_str": mmssms(float(t)), "level": int(lv), "symbol": symbols[int(lv)]}
        for t, lv in zip(times, levels)
    ]

def save_text(events: List[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Fireworks Code (time | symbol | level)\n")
        for ev in events:
            f.write(f"{ev['time_str']} | {ev['symbol']} | {ev['level']}\n")

def process_all(source_dir="music_source", out_dir="runs"):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(source_dir):
        if fname.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            in_path = os.path.join(source_dir, fname)
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{base}.txt")
            print(f"Processing {fname} -> {out_path}")
            events = encode(in_path)
            save_text(events, out_path)

if __name__ == "__main__":
    process_all()
