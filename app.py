"""
THE FORGETTING BOOTH — Flask server.
Records in the browser, runs QVAC locally, wipes everything after.
"""

import asyncio
import base64
import io
import os
import tempfile
import threading
import wave

import numpy as np
from flask import Flask, jsonify, render_template, request

from tetherto.qvac_sdk import (
    Client,
    load_model,
    completion,
    transcribe,
    text_to_speech,
    delete_cache,
)
from tetherto.qvac_sdk.schemas import TranscribeRequest, TextToSpeechRequest
from tetherto.qvac_sdk.models import (
    LLAMA_3_2_1B_INST_Q4_0,
    WHISPER_TINY,
    TTS_MULTILINGUAL_SUPERTONIC3_Q8_0,
)

PERSONA = (
    "You are The Void — a listener that receives and releases. "
    "You never judge and never advise unless explicitly asked. "
    "Reply in 2-3 short sentences. Be warm, spare, and a little strange. "
    "Never repeat what was said back verbatim."
)
DEFAULT_TTS_SR = 44100


def _pcm16_wav(samples, sr):
    # Per SDK: `buffer` is already signed 16-bit PCM in [-32768, 32767],
    # carried as Python floats. Do NOT re-scale — just clamp and cast.
    a = np.asarray(samples, dtype=np.float32)
    a = np.clip(a, -32768.0, 32767.0)
    pcm = a.astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


class Booth:
    """Owns the long-lived QVAC client and the loaded models."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.client = None
        self.models = {}
        self.lock = None
        self.ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.lock = asyncio.Lock()
        self.loop.run_forever()

    def bootstrap(self):
        asyncio.run_coroutine_threadsafe(self._bootstrap(), self.loop).result()
        self.ready.set()

    async def _bootstrap(self):
        self.client = Client()
        await self.client.__aenter__()
        t = self.client.transport
        self.models["asr"] = await load_model(
            t,
            model_src=WHISPER_TINY,
            model_config={"audio_format": "f32le", "language": "en"},
        )
        self.models["llm"] = await load_model(
            t, model_src=LLAMA_3_2_1B_INST_Q4_0
        )
        self.models["tts"] = await load_model(
            t,
            model_src=TTS_MULTILINGUAL_SUPERTONIC3_Q8_0,
            model_config={
                "ttsEngine": "supertonic",
                "language": "en",
                "voice": "F1",
                "ttsSpeed": 1.05,
                "ttsNumInferenceSteps": 5,
            },
        )

    def confess(self, wav_bytes):
        return asyncio.run_coroutine_threadsafe(
            self._confess(wav_bytes), self.loop
        ).result(timeout=180)

    async def _confess(self, wav_bytes):
        async with self.lock:
            t = self.client.transport

            # Write the uploaded WAV to a temp file, transcribe via filePath,
            # then delete it immediately.
            fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="booth_in_")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(wav_bytes)

                req = TranscribeRequest.model_validate(
                    {
                        "type": "transcribe",
                        "modelId": self.models["asr"],
                        "audioChunk": {"type": "filePath", "value": tmp_path},
                    }
                )
                secret = ""
                async for resp in transcribe(t, req):
                    if resp.error:
                        raise RuntimeError(resp.error)
                    if resp.text:
                        secret += resp.text
                secret = secret.strip()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if not secret:
                return {"transcript": "", "reply": "", "audio_b64": ""}

            # 2. completion
            run = completion(
                t,
                model_id=self.models["llm"],
                history=[
                    {"role": "system", "content": PERSONA},
                    {"role": "user", "content": secret},
                ],
            )
            reply = ""
            async for ev in run.events:
                if getattr(ev, "type", None) == "contentDelta":
                    reply += getattr(ev, "text", "") or ""
            if not reply.strip():
                reply = getattr(run, "text", "") or ""
            reply = reply.strip()

            # 3. text-to-speech
            tts_req = TextToSpeechRequest.model_validate(
                {
                    "type": "textToSpeech",
                    "modelId": self.models["tts"],
                    "text": reply,
                    "inputType": "text",
                    "stream": False,
                }
            )
            samples = []
            sr = DEFAULT_TTS_SR
            async for resp in text_to_speech(t, tts_req):
                if resp.sample_rate:
                    sr = resp.sample_rate
                if resp.buffer:
                    samples.extend(resp.buffer)

            audio_out_b64 = ""
            if samples:
                audio_out_b64 = base64.b64encode(
                    _pcm16_wav(samples, sr)
                ).decode("ascii")

            return {
                "transcript": secret,
                "reply": reply,
                "audio_b64": audio_out_b64,
            }

    def forget(self):
        return asyncio.run_coroutine_threadsafe(
            self._forget(), self.loop
        ).result(timeout=60)

    async def _forget(self):
        async with self.lock:
            t = self.client.transport
            wiped = True
            try:
                await delete_cache(t, auto=True)
            except Exception:
                wiped = False
            return {"ok": wiped}


booth = Booth()
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/booth", methods=["POST"])
def booth_endpoint():
    if "audio" not in request.files:
        return jsonify({"error": "no audio uploaded"}), 400
    if not booth.ready.wait(timeout=30):
        return jsonify({"error": "models still loading"}), 503
    audio_bytes = request.files["audio"].read()
    try:
        return jsonify(booth.confess(audio_bytes))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forget", methods=["POST"])
def forget_endpoint():
    try:
        return jsonify(booth.forget())
    except Exception as e:
        return jsonify({"error": str(e), "ok": False}), 500


if __name__ == "__main__":
    print("Loading QVAC models (first run downloads — be patient)...", flush=True)
    booth.bootstrap()
    print("\nReady.  ->  http://127.0.0.1:5000\n", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
