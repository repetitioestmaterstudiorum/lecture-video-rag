def remove_multiple_newlines(text: str):
    """
    Does only one thing: Ensures maximum of one newline character between lines.
    """
    lines = text.splitlines()

    clean_lines = []
    seen_blank = False
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if seen_blank:
                continue
            else:
                seen_blank = True
                clean_lines.append(line)
        else:
            seen_blank = False
            clean_lines.append(line)

    return '\n'.join(clean_lines)


def combine_sentences_overlapped(sentences, out_len: int, overlap=0, sentence_separators=['.', '!', '?', '\n']):
    """
    Takes a list of sentences, the desired output length of combined sentences, overlap, and returns combined sentences.

    `sentences` input format:
    {
        "text": "Hello World, this is CS50 and this is an introduction to artificial intelligence with Python with CS50's own Brian U.",
        "start": 0.0,
        "end": 29.44,
    }

    Combined sentences output format:
    {
        "text": "Hello World, this is CS50 and this is an introduction to artificial intelligence with Python with CS50's own Brian U. This course picks up where CS50 itself leaves off and explores the concepts and algorithms at the foundation of modern AI.",
        "start": 0.0,
        "end": 34.120000000000005,
        "len": 240
    }
    """
    if overlap >= out_len:
        print("Overlap must be smaller or equal to out_len.")
        overlap = out_len

    max_sentence_len = max(len(sentence['text'])
                           for sentence in sentences) + 1 + overlap  # + 1 for the space between sentences
    if out_len < max_sentence_len:
        print(
            f"Some sentences plus 'overlap' of {overlap} are larger than desired 'out_len' {out_len}. Setting new 'out_len': {max_sentence_len}")
        out_len = max_sentence_len

    combined_sentences = []

    sentence_separators_with_space = [
        f'{separator} ' for separator in sentence_separators]
    start = sentences[0]['start']
    accumulated_text = ''
    for i in range(0, len(sentences)):
        accumulated_text = ' '.join(
            [accumulated_text, sentences[i]['text']]).strip()

        next_sentence = sentences[i + 1]['text'] if i + \
            1 < len(sentences) else ''

        if len(accumulated_text) <= out_len and ((len(' '.join([accumulated_text, next_sentence])) + overlap + 1 >= out_len) or (i == len(sentences) - 1 and len(accumulated_text) > 0)):
            overlap_left = combined_sentences[-1]['text'][-overlap:
                                                          ] if overlap > 0 and combined_sentences else ''
            if overlap_left:
                try:
                    first_separator_index = min(overlap_left.find(
                        separator) for separator in sentence_separators_with_space if separator in overlap_left)
                    if first_separator_index != -1:
                        overlap_left = overlap_left[first_separator_index + 1:]
                except ValueError:
                    overlap_left = ''

            text_to_add = ' '.join([overlap_left, accumulated_text]).strip()
            combined_sentences.append({
                'text': text_to_add,
                'start': start,
                'end': sentences[i]['end'],
                'len': len(text_to_add),
            })

            start = sentences[i]['start']
            accumulated_text = ''

    return combined_sentences


def load_text(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()
