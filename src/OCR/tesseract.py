import cv2
import pytesseract
from PIL import Image
import numpy as np
from utils.text import remove_multiple_newlines

# ---


def video_to_text(video_file_path: str, capture_every_n_seconds: int, frame_diff_threshold: int, device='cpu', debug=False):
    if device != 'cpu':
        print(f"Only device 'cpu' is officially supported in pytesseract (March 2024). Switching to 'cpu'.")
        device = 'cpu'

    debug and print(
        f"Extracting video image every {capture_every_n_seconds} seconds")
    debug and print(f"Frame difference threshold: {frame_diff_threshold}%")

    video_capture = cv2.VideoCapture(video_file_path)

    assert video_capture.isOpened()

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    debug and print(f"Video frame rate: {fps}fps")

    frame_interval = int(capture_every_n_seconds * fps)

    output = []

    prev_frame = None
    frame_number = 0
    last_text_change_frame = 0
    last_text = None

    while True:
        # Jump to relevant frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        frame_returned, frame = video_capture.read()

        if not frame_returned:
            break

        # First frame: Log video dimensions, save first frame and continue
        if frame_number == 0:
            image_height, image_width, _ = frame.shape
            debug and print(
                f"Video dimensions: {image_width}x{image_height}px")
            debug and print('---')
            prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_number += frame_interval
            continue

        timestamp = int(frame_number / fps)

        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_frame, current_frame)
        diff_pct = round((np.count_nonzero(diff) * 100) / diff.size, 2)

        if diff_pct < frame_diff_threshold:
            debug and print(
                f"Frame difference below threshold at timestamp {timestamp}: {diff_pct}%. Skipping...")
            frame_number += frame_interval
            continue

        # OCR
        image = Image.fromarray(current_frame)
        text = pytesseract.image_to_string(image, lang='eng')
        text = remove_multiple_newlines(text)

        if not text:
            debug and print(
                f"No text detected at timestamp {timestamp}. Skipping...")
            frame_number += frame_interval
            continue

        if last_text is None or text != last_text:
            output.append({
                'text': text,
                'start': int(last_text_change_frame / fps),
                'end': timestamp,
                'len': len(text),
                'diff_pct': diff_pct
            })
            last_text_change_frame = frame_number
            last_text = text

        prev_frame = current_frame
        frame_number += frame_interval

    video_capture.release()

    # Add the last detected slide
    if last_text is not None:
        output.append({
            'text': last_text,
            'start': int(last_text_change_frame / fps),
            'end': int(frame_number / fps),
            'len': len(last_text),
        })

    return output
