# AI Doctor with Vision, Text, and Voice

This is a Gradio-based AI doctor demo that accepts text, voice, and an optional medical image. It can transcribe patient speech, analyze text/image input with Groq, and generate a spoken doctor response with ElevenLabs.

This project is for learning and prototyping only. It is not a medical device and should not be used as a replacement for a licensed clinician.

## Features

- Voice input from microphone through Gradio.
- Speech-to-text using Groq Whisper.
- Text and image analysis using Groq chat/vision.
- Image compression before sending to the model.
- Text-to-speech using ElevenLabs.
- Local `.env` loading through `config.py`.
- Unique generated audio filenames to avoid overwriting previous responses.

## Project Structure

```text
AI_DOCTOR_2.0/
  brain_of_the_doctor.py      # Groq vision/text analysis and image compression
  voice_of_the_patient.py     # Groq speech-to-text
  voice_of_the_doctor.py      # ElevenLabs/gTTS text-to-speech helpers
  gradio_app.py               # Main Gradio app
  config.py                   # Local .env loader and required env checks
  Pipfile                     # Python dependencies
  Pipfile.lock                # Locked dependency versions
  .gitignore                  # Files excluded from Git
```

## Requirements

- Python 3.12
- A Groq API key
- An ElevenLabs API key
- An ElevenLabs voice ID from your account
- ffmpeg installed and available on PATH for MP3 to WAV conversion through pydub

The existing virtual environment is in `venv/`, but for a clean setup you can recreate dependencies with Pipenv.

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id_here
```

Example voice IDs from your ElevenLabs account:

```text
Sarah - Mature, Reassuring, Confident: EXAVITQu4vr4xnSDxMaL
Matilda - Knowledgable, Professional: XrExE9yKIg1WjnnlVkGX
Devi - Clear Hindi pronunciation: MF4J4IDTRo0AxOO4dpFR
```

Do not commit `.env` to GitHub.

## Install

Using the existing virtual environment:

```powershell
cd C:\Users\marut\Downloads\MY-PROJECTS\AI_DOCTOR_2.0
.\venv\Scripts\Activate.ps1
```

Or using Pipenv:

```powershell
pipenv install
pipenv shell
```

If activation is blocked by PowerShell policy, run the app directly with:

```powershell
.\venv\Scripts\python.exe gradio_app.py
```

## Run

```powershell
python gradio_app.py
```

Or:

```powershell
.\venv\Scripts\python.exe gradio_app.py
```

The app starts at a local Gradio URL such as:

```text
http://127.0.0.1:7860
```

If port `7860` is busy, Gradio may choose another port such as `7861`.

## How The App Works

1. The user provides text, voice, and optionally an image.
2. If voice is provided, `voice_of_the_patient.py` sends it to Groq Whisper:

```python
stt_model = "whisper-large-v3-turbo"
```

3. If an image is provided, `brain_of_the_doctor.py` compresses it before base64 encoding:

```python
def encode_image(image_path, max_size=(1024, 1024), quality=75):
```

4. Groq analyzes the prompt and optional image using:

```python
model="meta-llama/llama-4-scout-17b-16e-instruct"
```

5. ElevenLabs generates a doctor voice response using:

```python
model="eleven_flash_v2"
voice=require_env("ELEVENLABS_VOICE_ID")
```

6. The MP3 is converted to WAV for Gradio playback.

## Important Modifications Made

### Environment Loading

Added `config.py` so Python files can load `.env` directly:

```python
load_env_file()
require_env("GROQ_API_KEY")
require_env("ELEVENLABS_API_KEY")
require_env("ELEVENLABS_VOICE_ID")
```

This avoids failures when VS Code terminal environment injection is disabled.

### Groq API Key Fix

Earlier failure:

```text
The api_key client option must be set
```

Cause: `.env` existed but was not loaded into the running Python process.

Fix: `config.py` now loads `.env`, and Groq clients use `require_env()`.

### Model Updates

Updated speech-to-text model:

```python
whisper-large-v3-turbo
```

Updated ElevenLabs model:

```python
eleven_flash_v2
```

The older hardcoded ElevenLabs voice name `Aria` was replaced with:

```python
ELEVENLABS_VOICE_ID
```

### Image Compression

Large uploaded images are compressed before being sent to Groq. This improves speed and helps avoid request size limits/timeouts.

Current compression settings:

```python
max_size=(1024, 1024)
quality=75
```

For faster but lower-detail requests:

```python
max_size=(768, 768)
quality=70
```

For better image detail:

```python
max_size=(1280, 1280)
quality=80
```

### Text-Only Mode

Text-only input now goes through the same Groq chat path instead of returning only a hardcoded response.

### Unique Audio Output

Generated audio files now use unique names:

```python
final_<uuid>.mp3
final_<uuid>.wav
```

This avoids overwriting `final.mp3` and `final.wav` on repeated requests.

## Common Errors And Fixes

### Missing Groq API Key

Error:

```text
The api_key client option must be set
```

Fix:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Restart the app after editing `.env`.

### ElevenLabs Voice Not Found

Error:

```text
Voice Aria not found
```

Fix: use a valid voice ID from your ElevenLabs account:

```env
ELEVENLABS_VOICE_ID=XrExE9yKIg1WjnnlVkGX
```

### Request Timed Out

Likely cause:

- Large image upload
- Temporary API slowdown
- Network instability
- Long model response

Fixes:

- Keep image compression enabled.
- Reduce image size to `(768, 768)` if needed.
- Limit model output with `max_tokens`.
- Consider adding fallback logic to another vision model provider.

### Too Little Data For Declared Content-Length

This can happen while Gradio serves generated audio files.

Possible fixes:

- Return MP3 directly instead of converting to WAV.
- Ensure generated files are not deleted or overwritten while Gradio serves them.
- Keep unique filenames enabled.

### ffmpeg Missing

If MP3 to WAV conversion fails, install ffmpeg and make sure it is available on PATH.

## Git And GitHub

This repo should not commit local secrets or generated files.

Current `.gitignore` should include:

```gitignore
.env
venv/
__pycache__/
.gradio/
*.mp3
*.wav
```

To push a new branch:

```powershell
git checkout -b update-ai-doctor
git add .
git commit -m "Update AI doctor app"
git push -u origin update-ai-doctor
```

Then open a Pull Request on GitHub and merge it into `main`.

Remote repository:

```text
https://github.com/maruthikelgire/AI-DOCTOR-with-vision-text-and-voice.git
```

## Future Modifications

- Return MP3 directly in Gradio and remove WAV conversion to reduce file-serving issues.
- Add `max_tokens`, `temperature`, and a timeout to Groq chat requests.
- Add fallback provider support, such as Gemini Flash/Flash-Lite, if Groq times out.
- Move model names into `.env` so they can be changed without editing code.
- Add `.env.example` with safe placeholder values.
- Add a cleanup job for old generated `final_*.mp3` and `final_*.wav` files.
- Add a proper `requirements.txt` for users who do not use Pipenv.
- Add better medical safety wording and a clinician disclaimer in the UI.
- Add tests for `.env` loading, image compression, and missing API keys.
- Add user-selectable voice options in the Gradio interface.
- Improve text-only mode with a dedicated system prompt.

## Development Notes

Before committing:

```powershell
git status
```

Make sure these are not staged:

```text
.env
venv/
*.mp3
*.wav
.gradio/
```

Quick syntax check:

```powershell
.\venv\Scripts\python.exe -m py_compile config.py brain_of_the_doctor.py voice_of_the_patient.py voice_of_the_doctor.py gradio_app.py
```
