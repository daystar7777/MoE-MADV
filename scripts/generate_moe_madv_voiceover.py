#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "docs" / "video" / "build"
VOICEOVER_TXT = BUILD / "moe-madv-shorts-voiceover.txt"
VOICEOVER_MP3 = BUILD / "moe-madv-shorts-voiceover.mp3"


VOICEOVER = """This is DeepSeek V4 Flash, a 284-billion-parameter Mixture-of-Experts model.
The GGUF file is about 150 gigabytes, and this run is on a 64 gigabyte M1 Max.

The system specs, elapsed time, memory, and I/O are visible while the local model generates tokens.

The slow part is not just compute. For this MoE model, each token can route to a different expert set, so the runtime has to bring the right expert pages into memory at the right time.

MoE-MADV gives macOS a MADV_WILLNEED hint after routing chooses the active experts.

On the decode benchmark, generation improved from 0.98 to 1.23 tokens per second: a 25.4 percent throughput gain without changing the model.

Scripts, data, charts, and reproduction notes are on GitHub at daystar7777 slash MoE-MADV.
"""


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    for path in [ROOT / ".env", ROOT.parent / ".env"]:
        if not path.exists():
            continue
        for raw in path.read_text(errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def request_json(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def choose_voice(voices: list[dict], preferred_id: str | None = None) -> tuple[str, str]:
    if preferred_id:
        for voice in voices:
            if voice.get("voice_id") == preferred_id:
                return preferred_id, voice.get("name", preferred_id)
        return preferred_id, preferred_id

    preferred_names = [
        "River - Relaxed, Neutral, Informative",
        "Alice - Clear, Engaging Educator",
        "Daniel - Steady Broadcaster",
        "Brian - Deep, Resonant and Comforting",
    ]
    by_name = {voice.get("name"): voice for voice in voices}
    for name in preferred_names:
        voice = by_name.get(name)
        if voice:
            return voice["voice_id"], name

    for voice in voices:
        labels = voice.get("labels") or {}
        if labels.get("use_case") == "informative_educational" and labels.get("language") == "en":
            return voice["voice_id"], voice.get("name", voice["voice_id"])

    if not voices:
        raise RuntimeError("No ElevenLabs voices returned for this account.")
    first = voices[0]
    return first["voice_id"], first.get("name", first["voice_id"])


def generate_audio(api_key: str, voice_id: str, model_id: str) -> bytes:
    body = {
        "text": VOICEOVER,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.62,
            "similarity_boost": 0.78,
            "style": 0.10,
            "use_speaker_boost": True,
            "speed": 0.92,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
        data=data,
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    env = load_env()
    api_key = env.get("elevenlabs_api_key") or env.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Missing elevenlabs_api_key in .env")

    headers = {"xi-api-key": api_key}
    voices = request_json("https://api.elevenlabs.io/v1/voices", headers).get("voices", [])
    voice_id, voice_name = choose_voice(voices, env.get("ELEVENLABS_VOICE_ID"))
    model_id = env.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")

    VOICEOVER_TXT.write_text(VOICEOVER)
    audio = generate_audio(api_key, voice_id, model_id)
    VOICEOVER_MP3.write_bytes(audio)

    print(f"voice: {voice_name}")
    print(f"model: {model_id}")
    print(f"text:  {VOICEOVER_TXT}")
    print(f"audio: {VOICEOVER_MP3} ({len(audio) / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
