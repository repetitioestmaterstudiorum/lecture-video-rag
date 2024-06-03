import re

# ---


def save_text(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def get_sentences_from_whisper_result(whisper_out_segments, sentence_separators=['.', '!', '?']):
    """
    Takes a whisper model output and a list of sentence separators, based on which sentences are recognized.
    Returns a list of sentences as dicts in the following format:
    {
        "text": "Hello World, this is CS50 and this is an introduction to artificial intelligence with Python with CS50's own Brian U.",
        "start": 0.0,
        "end": 29.44,
    }
    """
    def contains_sentence_separator(txt: str):
        return any(sep in txt for sep in sentence_separators)

    def split_text(txt: str):
        # Separator followed by a whitespace character
        pattern = '|'.join(
            re.escape(sep) + '(?=\s)' for sep in sentence_separators)
        return re.split(f'({pattern})\s', txt)

    sentences = []

    start = whisper_out_segments[0]['start']
    text = ''
    for segment in whisper_out_segments:
        text += segment['text']
        if contains_sentence_separator(text):
            splits = split_text(text)
            for i in range(0, len(splits) - 1, 2):
                # sentence and sentence sep.
                sentence = (splits[i] + splits[i + 1]).strip()
                if sentence:
                    sentences.append({
                        'text': sentence,
                        'start': start,
                        'end': segment['end']
                    })
                start = segment['end']
            text = splits[-1].strip()
            start = segment['start']

    if text:
        sentences.append({
            'text': text,
            'start': start,
            'end': whisper_out_segments[-1]['end']
        })

    return sentences
