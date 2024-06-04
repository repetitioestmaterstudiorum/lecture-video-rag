import os
from RAG.storage import Storage
from RAG.retriever import Retriever
from RAG.generator import Generator
from LLM.llama_cpp_model import LlamaCppLlm
from utils.seeds import set_seeds
from sentence_transformers import CrossEncoder
from typing import List, Literal

#  ---


class RAGSystem:
    def __init__(
        self,
        hypers: dict,
        data_dir: str,
        random_seed: int = None,
    ):
        self.hypers = hypers

        RANDOM_SEED = set_seeds(random_seed) if random_seed else set_seeds()

        self.storage_llm = LlamaCppLlm(
            data_path=os.path.join(data_dir, 'llms'),
            model_name=hypers['storage_llm_model_name'],
            system_message=hypers['storage_llm_system_message'],
            random_seed=RANDOM_SEED,
        )
        self.storage = Storage(
            db_path=os.path.join(data_dir, 'storage_db'),
            embedding_model=hypers['storage_embedding_model'],
            collection_space=hypers['storage_collection_space'],
            asr_chunk_size=hypers['storage_asr_chunk_size'],
            asr_overlap=hypers['storage_asr_overlap'],
            asr_whisper_model_name=hypers['storage_asr_whisper_model_name'],
            asr_postprocessing_fn=hypers['storage_asr_postprocessing_fn'],
            ocr_chunk_size=hypers['storage_ocr_chunk_size'],
            ocr_overlap=hypers['storage_ocr_overlap'],
            ocr_capture_every_n_s=hypers['storage_ocr_capture_every_n_s'],
            ocr_frame_diff_threshold=hypers['storage_ocr_frame_diff_threshold'],
            ocr_library=hypers['storage_ocr_library'],
            ocr_postprocessing_fn=hypers['storage_ocr_postprocessing_fn'],
            ocr_llm_preprompt=hypers['storage_ocr_llm_preprompt'],
            average_asr_ocr=hypers['storage_average_asr_ocr'],
            llm=self.storage_llm,
            cache_dir=os.path.join(data_dir, 'storage_cache'),
        )

        retriever_llm_same_as_storage_llm = True if hypers['storage_llm_model_name'] == hypers[
            'retriever_llm_model_name'] and hypers['storage_llm_system_message'] == hypers['retriever_llm_system_message'] else False
        self.retriever_llm = self.storage_llm if retriever_llm_same_as_storage_llm else LlamaCppLlm(
            data_path=os.path.join(data_dir, 'llms'),
            model_name=hypers['retriever_llm_model_name'],
            system_message=hypers['retriever_llm_system_message'],
            random_seed=RANDOM_SEED,
        )

        reranker = CrossEncoder(hypers['retriever_reranker_model_name']
                                ) if hypers['retriever_reranker_model_name'] else None

        self.retriever = Retriever(
            storage=self.storage,
            llm=self.retriever_llm,
            reranker=reranker,
        )

        generator_llm_same_as_storage_llm = True if hypers['storage_llm_model_name'] == hypers[
            'generator_llm_model_name'] and hypers['storage_llm_system_message'] == hypers['generator_llm_system_message'] else False

        generator_llm_same_as_retriever_llm = True if hypers['retriever_llm_model_name'] == hypers[
            'generator_llm_model_name'] and hypers['retriever_llm_system_message'] == hypers['generator_llm_system_message'] else False

        self.generator_llm = self.storage_llm if generator_llm_same_as_storage_llm else self.retriever_llm if generator_llm_same_as_retriever_llm else LlamaCppLlm(
            data_path=os.path.join(data_dir, 'llms'),
            model_name=hypers['generator_llm_model_name'],
            system_message=hypers['generator_llm_system_message'],
            random_seed=RANDOM_SEED,
        )

        self.generator = Generator(
            retriever=self.retriever,
            llm=self.generator_llm,
        )

        self.default_n_docs = hypers['retriever_n_docs'] if hypers['retriever_n_docs'] is not None else None
        self.default_modalities = hypers['retriever_modalities_overwrite'] if hypers[
            'retriever_modalities_overwrite'] is not None else None
        self.default_distance_threshold = hypers['retriever_distance_threshold'] if hypers[
            'retriever_distance_threshold'] is not None else None
        self.default_context_preparation_scheme = hypers['retriever_context_preparation_scheme'] if hypers[
            'retriever_context_preparation_scheme'] is not None else None
        self.default_reranker_n_docs = hypers['retriever_reranker_n_docs'] if hypers[
            'retriever_reranker_n_docs'] is not None else None
        self.default_generator_preprompt = hypers['generator_preprompt'] if hypers[
            'generator_preprompt'] is not None else None
        self.default_gen_max_tokens = hypers['generator_max_tokens'] if hypers[
            'generator_max_tokens'] is not None else None
        self.default_gen_temperature = hypers['generator_temperature'] if hypers[
            'generator_temperature'] is not None else None

    def ask(
        self,
        question: str,
        context_details: bool = True,
        force_answer: bool = False,
        context_detail_top_k: int | None = None,
        debug: bool = False,

        # Hyperparameters
        n_docs: int | None = None,
        modalities: List[Literal['asr', 'ocr', 'asr-ocr-avg']
                         ] | None = None,
        distance_threshold: int | None = None,
        context_preparation_scheme: Literal['add_text_descriptions',
                                            'llm_filter_relevant'] | None = None,
        reranker_n_docs: int | None = None,
        generator_preprompt: str | None = None,
        gen_max_tokens: int | None = None,
        gen_temperature: float | None = None,
    ):
        return self.generator.generate(
            question=question,
            n_docs=n_docs if n_docs is not None else self.default_n_docs,
            modalities=modalities if modalities is not None else self.default_modalities,
            distance_threshold=distance_threshold if distance_threshold is not None else self.default_distance_threshold,
            force_answer=force_answer,
            context_preparation_scheme=context_preparation_scheme if context_preparation_scheme is not None else self.default_context_preparation_scheme,
            reranker_n_docs=reranker_n_docs if reranker_n_docs is not None else self.default_reranker_n_docs,
            generator_preprompt=generator_preprompt if generator_preprompt is not None else self.default_generator_preprompt,
            gen_max_tokens=gen_max_tokens if gen_max_tokens is not None else self.default_gen_max_tokens,
            gen_temperature=gen_temperature if gen_temperature is not None else self.default_gen_temperature,
            context_details=context_details,
            context_detail_top_k=context_detail_top_k,
            debug=debug
        )
