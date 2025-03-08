README (README.md)
markdown
Copy
Edit
# Meeting Transcription Generator (Whisper)

## Overview
This project provides an automatic transcription service using OpenAI's Whisper model. Users can upload an audio file, and the system will transcribe the content and send the transcription via email. It is built using Gradio for a user-friendly web interface.

## Features
- Upload audio files for transcription.
- Uses OpenAI's Whisper model for accurate transcription.
- Automatically sends the transcription to the user's email.
- Built with Gradio for an intuitive user experience.

## Requirements
Make sure you have the following dependencies installed before running the application:

```bash
pip install gradio whisper langchain opensearch-py sentence-transformers python-docx

```
Usage
Run the application:
```bash
python whisper_console_email.py
```
Open the web interface in your browser.
Upload an audio file and enter your email address.
The transcription will be processed and emailed to you.
Configuration
Modify the send_email_with_attachment function to set up your email:

python
Copy
Edit
from_email = "your_email@gmail.com"
email_password = "your_email_password"
For security reasons, consider using an app-specific password instead of your main email password.

Notes
The model runs on a CUDA-enabled GPU for better performance.
Temporary files are stored and managed automatically to prevent storage issues.