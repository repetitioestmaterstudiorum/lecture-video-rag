import PyPDF2

# ---


def combine_sequences(sentences):
    """
    Takes a list of sentences and returns all combined sentences' text.

    `sentences` input format:
    {
        "text": "Hello World, this is CS50 and this is an introduction to artificial intelligence with Python with CS50's own Brian U.",
        "start": 0.0,
        "end": 29.44,
    }

    Combined sentences output format:
    {
        'text': 'Hello World, this is CS50 and this is an introduction to artificial intelligence with Python with CS50's own Brian U. ...',
        'start': 0,
        'end': 40,
        'len': 100
    }
    """
    if sentences == None or len(sentences) == 0:
        return None

    text = ''

    for i in range(0, len(sentences)):
        text = ' '.join(
            filter(None, [text, sentences[i]['text']]))

    return {
        "text": text,
        "start": sentences[0]['start'],
        "end": sentences[-1]['end'],
        "len": len(text)
    }


def extract_text_from_pdf(pdf_file_path: str):
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n"

    return extracted_text
