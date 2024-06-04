import asyncio
import threading
import sqlite3
import json
import os
import numpy as np
from OCR import easyocr, tesseract
from OCR.utils import combine_sequences
from ASR import whisp
from ASR.utils import get_sentences_from_whisper_result
from utils.text import combine_sentences_overlapped
from concurrent.futures import Future
from DB.chroma import DB
from typing import Literal
from utils.torch import detect_gpu
from LLM.llama_cpp_model import LlamaCppLlm
from utils.hash import hash_256, hash_256_tob, encode_base64

#  ---

ocr_postprocessing_fns = {}

asr_postprocessing_fns = {
    'get_sentences_from_whisper_result': get_sentences_from_whisper_result
}


def cache_method(method):
    def wrapper(self, *args, **kwargs):
        cache_key = self._generate_cache_key(method.__name__, *args, **kwargs)
        db_path = os.path.join(self.cache_dir, 'asr-ocr-cache.db')

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT result FROM cache WHERE key = ?', (cache_key,))
            result = cursor.fetchone()

            if result:
                print(f"Cached hit for {method.__name__}!")
                return json.loads(result[0])
            else:
                print(f"No cache hit for {method.__name__}. Running...")
                result = method(self, *args, **kwargs)
                cursor.execute('INSERT INTO cache (key, result) VALUES (?, ?)',
                               (cache_key, json.dumps(result)))
                conn.commit()
                return result
    return wrapper


class Storage:
    def __init__(
        self,
        db_path: str = None,
        embedding_model: str = 'nomic-ai/nomic-embed-text-v1',
        collection_space: Literal['cosine', 'l2', 'ip'] = 'cosine',
        asr_chunk_size: int = 1000,
        asr_overlap: int = 200,
        asr_whisper_model_name: str = 'tiny.en',
        asr_postprocessing_fn: str | None = None,
        ocr_chunk_size: int = 300,
        ocr_overlap: int = 50,
        ocr_capture_every_n_s: int = 10,
        ocr_frame_diff_threshold: int = 15,
        ocr_library: Literal['easyocr', 'tesseract'] = 'easyocr',
        ocr_postprocessing_fn: str | None = None,
        ocr_llm_preprompt: str = "",
        average_asr_ocr: bool = False,
        llm: LlamaCppLlm | None = None,
        cache_dir: str | None = None,
        device: str = detect_gpu(),
    ):
        self.db = DB(db_path)
        self.asr_chunk_size = asr_chunk_size
        self.asr_overlap = asr_overlap
        self.asr_whisper_model_name = asr_whisper_model_name
        self.asr_postprocessing_fn = asr_postprocessing_fn
        self.ocr_chunk_size = ocr_chunk_size
        self.ocr_overlap = ocr_overlap
        self.ocr_capture_every_n_s = ocr_capture_every_n_s
        self.ocr_frame_diff_threshold = ocr_frame_diff_threshold
        self.ocr_postprocessing_fn = ocr_postprocessing_fn
        self.ocr_llm_preprompt = ocr_llm_preprompt
        self.llm = llm
        self.ocr_library = ocr_library
        self.average_asr_ocr = average_asr_ocr
        self.cache_dir = cache_dir
        self.device = device

        if self.ocr_llm_preprompt and self.llm == None:
            raise TypeError(
                f"An LLM is required when using 'ocr_llm_preprompt'.")

        # Calculate the collection name from arguments
        storage_config = locals()
        storage_config.pop('self')
        storage_config.pop('db_path')
        storage_config.pop('llm')
        if llm:
            storage_config['llm.model_name'] = llm.model_name
        storage_config.pop('device')
        sc_hash = hash_256_tob(storage_config)
        # Collection name max len: 63, therefore base64 encoding
        sc_hash_base64 = encode_base64(sc_hash)
        self.collection_name = f"lecture-videos-{sc_hash_base64.rstrip('=')}"
        print(f"Collection name: {self.collection_name}")
        self.collection = self.db.get_collection(
            self.collection_name, space=collection_space)

        if embedding_model in ['nomic-ai/nomic-embed-text-v1', 'nomic-ai/nomic-embed-text-v1.5']:
            from embed import nomic
            self.embedder = nomic.Embedder(
                hf_model_name=embedding_model, device=device)
            print(f"Initalized embedding model: {embedding_model}")
        else:
            raise ValueError(
                f"Not implemented embedding model: {embedding_model}")

        self._initialize_cache()

    def add_video(
        self,
        unique_video_name: str,
        video_path: str,
        parallel: bool = True,
        debug: bool = False
    ):
        if not unique_video_name or not video_path:
            raise ValueError(
                "Both 'unique_video_name' and 'video_path' need to be provided.")

        if self.get_is_video_in_db(unique_video_name):
            debug and print(
                f"Video '{unique_video_name}' is already in the DB. Skipping.")
            return
        else:
            debug and print(f"Adding video '{unique_video_name}'...")

        debug and print("Preprocessing...")
        if parallel:
            future = Future()
            coroutine = self._process_video_async(video_path, debug)
            threading.Thread(target=run_async_coroutine,
                             args=(coroutine, future)).start()
            asr_out, ocr_out = future.result()
        else:
            asr_out = self._asr_video_to_text(video_path, debug)
            ocr_out = self._ocr_video_to_text(video_path, debug)
        print(f"\nASR segments: {len(asr_out)}, OCR segments: {len(ocr_out)}")

        # Postprocessing
        if self.asr_postprocessing_fn:
            asr_out = asr_postprocessing_fns[self.asr_postprocessing_fn](
                asr_out)

        asr_sequences = combine_sentences_overlapped(
            asr_out, out_len=self.asr_chunk_size, overlap=self.asr_overlap)

        if self.ocr_postprocessing_fn:
            ocr_out = ocr_postprocessing_fns[self.ocr_postprocessing_fn](
                ocr_out)

        if self.ocr_llm_preprompt and self.llm:
            debug and print(f"Cleaning OCR text with LLM...")
            for seq in ocr_out:
                seq['text'] = self._clean_ocr_text_with_llm(seq['text'])

        ocr_sequences = combine_sentences_overlapped(
            sentences=ocr_out, out_len=self.ocr_chunk_size, overlap=self.ocr_overlap)

        # Embedding and inserting into vector DB
        debug and print("Embedding and inserting into vector DB...")
        if self.average_asr_ocr:
            asr_ocr_avg_sequences = []
            last_used_ocr_index = 0
            for asr_sequence in asr_sequences:
                debug and print(
                    f"ASR start: {asr_sequence['start']}, ASR end: {asr_sequence['end']}")
                relevant_ocr_seqs = []
                # Start from the last used OCR index
                for index in range(last_used_ocr_index, len(ocr_sequences)):
                    ocr_sequence = ocr_sequences[index]
                    if ocr_sequence['end'] < asr_sequence['start']:
                        # OCR seq ends before the ASR seq starts, skip it
                        last_used_ocr_index = index + 1
                    elif ocr_sequence['start'] > asr_sequence['end']:
                        # OCR seq starts after the ASR seq ends, stop the search
                        break
                    else:
                        # OCR seq overlaps with the ASR seq
                        relevant_ocr_seqs.append(ocr_sequence)
                debug and print(
                    f"Relevant OCR sequences: {len(relevant_ocr_seqs)}")
                ocr_sequence_comb = combine_sequences(relevant_ocr_seqs)

                asr_ocr_avg_sequences.append({
                    'start': min(asr_sequence['start'], ocr_sequence_comb['start']) if ocr_sequence_comb else asr_sequence['start'],
                    'end': max(asr_sequence['end'], ocr_sequence_comb['end']) if ocr_sequence_comb else asr_sequence['end'],
                    'asr_text': asr_sequence['text'],
                    'ocr_text': ocr_sequence_comb['text'],
                    'ocr_else_asr_text': ocr_sequence_comb['text'] if ocr_sequence_comb else asr_sequence['text'],
                })

            self._embed_insert_sequences(
                asr_ocr_avg_sequences, 'asr-ocr-avg', unique_video_name)
        else:
            self._embed_insert_sequences(
                asr_sequences, 'asr', unique_video_name)
            self._embed_insert_sequences(
                ocr_sequences, 'ocr', unique_video_name)

    def remove_video(self, unique_video_name: str):
        matched_docs = self.collection.get(
            where={'unique_video_name': unique_video_name})

        if matched_docs and 'ids' in matched_docs and matched_docs['ids']:
            doc_ids_to_delete = matched_docs['ids']

            deletion_result = self.collection.delete(ids=doc_ids_to_delete)

            return deletion_result
        else:
            print("No documents found for the given video name.")
            return

    def get_is_video_in_db(self, unique_video_name: str):
        unique_video_name_doc = self.collection.get(
            where={'unique_video_name': unique_video_name}
        )

        n_documents_with_name = len(unique_video_name_doc.get('ids', []))

        if n_documents_with_name > 0:
            return True
        else:
            return False

    @cache_method
    def _asr_video_to_text(self, video_path: str, debug=False):
        asr_output = whisp.video_to_text(
            video_file_path=video_path,
            model_name=self.asr_whisper_model_name,
            device=self.device
        )
        debug and print(f"ASR done.")
        return asr_output['segments']

    @cache_method
    def _ocr_video_to_text(self, video_path: str, debug=False):
        ocr_args = {
            'video_file_path': video_path,
            'capture_every_n_seconds': self.ocr_capture_every_n_s,
            'frame_diff_threshold': self.ocr_frame_diff_threshold,
            'device': self.device,
            'debug': debug
        }
        if self.ocr_library == 'easyocr':
            ocr_output = easyocr.video_to_text(**ocr_args)
        elif self.ocr_library == 'tesseract':
            ocr_output = tesseract.video_to_text(**ocr_args)
        else:
            raise ValueError(
                f"Unsupported ocr library name: {self.ocr_library}")

        debug and print(f"OCR done.")
        return ocr_output

    def _embed_insert_sequences(self, sequences, source: str, unique_video_name: str):
        if source == 'asr-ocr-avg':
            print("Averaging ASR and OCR sequences.")
            asr_texts = list(map(lambda s: s['asr_text'], sequences))
            asr_embeddings = self.embedder.embed_documents(asr_texts)
            asr_embeddings = [np.array(e) for e in asr_embeddings]
            ocr_else_asr_texts = list(
                map(lambda s: s['ocr_else_asr_text'], sequences))
            ocr_else_asr_embeddings = self.embedder.embed_documents(
                ocr_else_asr_texts)
            ocr_else_asr_embeddings = [
                np.array(e) for e in ocr_else_asr_embeddings]
            embeddings = []
            for asr_e, ocr_e in zip(asr_embeddings, ocr_else_asr_embeddings):
                combo = (asr_e + ocr_e) / 2
                embeddings.append(combo.tolist())
            texts = [f"{s['asr_text']}\n{s['ocr_text']}" if s['ocr_text']
                     else s['asr_text'] for s in sequences]

        else:
            texts = list(map(lambda s: s['text'], sequences))
            embeddings = self.embedder.embed_documents(texts)

        metadatas = list(map(lambda s: {
            'unique_video_name': unique_video_name,
            'source': source,
            'start': s['start'],
            'end': s['end'],
        }, sequences))

        self.collection.insert(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )

    async def _process_video_async(self, video_path: str, debug=False):
        asr_future = asyncio.to_thread(
            self._asr_video_to_text, video_path, debug)
        ocr_future = asyncio.to_thread(
            self._ocr_video_to_text, video_path, debug)
        return await asyncio.gather(asr_future, ocr_future)

    def _clean_ocr_text_with_llm(self, ocr_text: str):
        prompt = (f"OCR text:\n{ocr_text}\n\n{self.ocr_llm_preprompt}")
        # 1 token is appr. 4 characters, so twice the length of the OCR text (https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
        max_tokens = len(ocr_text) / 2
        ocr_text_cleaned = self.llm.generate(prompt, max_tokens)
        return ocr_text_cleaned

    def _initialize_cache(self):
        if not self.cache_dir:
            self.cache_dir = os.path.join(os.getcwd(), 'asr-ocr-cache')

        print("Initializing ASR and OCR cache...")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        db_path = os.path.join(self.cache_dir, 'asr-ocr-cache.db')
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS cache (
                                key TEXT PRIMARY KEY,
                                result TEXT)''')
            conn.commit()

    def _generate_cache_key(self, method_name, *args, **kwargs):
        hash_input = method_name + str(args) + str(kwargs)
        for attr in [
            'asr_whisper_model_name',
            'ocr_library',
            'ocr_capture_every_n_s',
            'ocr_frame_diff_threshold'
        ]:
            if hasattr(self, attr):
                hash_input += str(getattr(self, attr))
        return hash_256(hash_input)


def run_async_coroutine(coroutine, future: Future):
    try:
        # Create a new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coroutine)
        loop.close()
        # Set the result of the future to the result of the coroutine
        future.set_result(result)
    except Exception as e:
        future.set_exception(e)
