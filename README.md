# The Forgetting Booth

An on-device AI confessional. Speak a secret. It's transcribed locally,
answered locally, and spoken back locally. Then everything is provably
deleted — the audio, the models from RAM, and the worker cache — and you
receive a Certificate of Forgetting.

Nothing leaves your machine. No API keys. No cloud. Turn off WiFi and it
still works.

## What it does

1. Browser records 8 seconds of audio (16 kHz WAV, in memory).
2. Sent to a Flask server running on `127.0.0.1` only.
3. QVAC transcribes it locally with Whisper.
4. Llama 3.2 1B replies as "The Void" — a listener that receives and releases.
5. Supertonic TTS speaks the reply back.
6. Audio file, models, and cache are wiped. Certificate is issued.

## QVAC functions called

- `load_model` — Whisper Tiny (ASR), Llama 3.2 1B (LLM), Supertonic 3 (TTS)
- `transcribe` — speech to text, on-device
- `completion` — LLM reply, on-device
- `text_to_speech` — reply to speech, on-device
- `delete_cache` — wipes worker cache
- `unload_model` — frees RAM

## SDK version used

`tetherto-qvac-sdk >= 0.19.0`

## System requirements

- macOS 14+ (Apple Silicon) / Linux Ubuntu 22+ / Windows 10+ (Vulkan required)
- Node.js >= 22.17 (required — the QVAC worker runs on Node)
- Python >= 3.10
- ~2 GB free disk for model downloads (first run only)
- A microphone

## Install

Clone the repo, then from the project root:

    python3 -m venv venv
    source venv/bin/activate

On Windows use `venv\Scripts\activate` instead.

Install the QVAC SDK (fat wheel with the bundled worker):

    pip install "tetherto-qvac-sdk>=0.19.0" -f https://github.com/tetherto/qvac/releases/expanded_assets/sdk-v0.20.0

Then the rest of the dependencies:

    pip install -r requirements.txt

If `Client()` later raises `WorkerNotFoundError`, the fat wheel for your
platform wasn't picked up — reinstall with the `-f` URL above and confirm:

    python -c "from tetherto.qvac_sdk import Client; Client(); print('worker OK')"

## Run

    python app.py

First run downloads ~1 GB across three models (Whisper Tiny, Llama 3.2 1B,
Supertonic 3). Wait for the line:

    Ready.  ->  http://127.0.0.1:5000

Open http://127.0.0.1:5000 in your browser, allow microphone access, and
click **confess**.

## Proof of on-device

Turn off WiFi. Run it. Speak. It still works. Nothing is uploaded.

## License

MIT — see `LICENSE`.
