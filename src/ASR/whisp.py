from whisper import load_audio, load_model

# ---


def video_to_text(video_file_path: str, model_name="tiny.en", device='cpu', debug=False):
    if device == 'mps':
        debug and print(
            f"The device 'mps' is not supported in whisper (March 2024). Switching to 'cpu'. ")
        device = 'cpu'

    model = load_model(model_name, device=device)

    audio_file = load_audio(video_file_path, sr=16000)

    result = model.transcribe(
        audio_file,
        fp16=False,
    )

    return result
