from RAG.retriever import Retriever
from typing import List, Literal, Any
from LLM.llama_cpp_model import LlamaCppLlm

# ---

DEFAULT_DISTANCE_THRESHOLD = 1


class Generator:
    def __init__(self, retriever: Retriever, llm: LlamaCppLlm):
        self.retriever = retriever
        self.llm = llm

    def generate(
        self,
        question: str,
        n_docs: int = 5,
        modalities: List[Literal['asr', 'ocr', 'asr-ocr-avg']
                         ] = ['asr', 'ocr', 'asr-ocr-avg'],
        distance_threshold: int = DEFAULT_DISTANCE_THRESHOLD,
        force_answer: bool = False,
        context_preparation_scheme: Literal['add_text_descriptions',
                                            'llm_filter_relevant'] | None = None,
        reranker_n_docs: int = 30,
        generator_preprompt: str = '',
        gen_max_tokens: int = 512,
        gen_temperature: float = 0.7,
        context_details: bool = True,
        context_detail_top_k: int | None = None,
        stream: bool = False,
        debug: bool = False
    ):
        if not context_detail_top_k:
            context_detail_top_k = n_docs
            debug and print(
                f"Set context_detail_top_k to n_docs: {n_docs}.")
        if context_detail_top_k > n_docs:
            context_detail_top_k = n_docs
            debug and print(
                f"context_detail_top_k cannot be greater than n_docs. Set context_detail_top_k to {n_docs}.")

        if not modalities:
            debug and print("No modalities provided. Using all modalities.")
            modalities = ['asr', 'ocr', 'asr-ocr-avg']

        metadatas, distances, documents = self.retriever.get_context_data(
            question=question,
            n_docs=n_docs,
            modalities=modalities,
            distance_threshold=distance_threshold,
            context_preparation_scheme=context_preparation_scheme,
            reranker_n_docs=reranker_n_docs,
            debug=debug
        )

        if debug and documents:
            for meta, dist, doc in zip(metadatas, distances, documents):
                print(
                    f"Dist: {dist:.2f}, source: {meta.get('source')}, doc: {doc[:50]}...")

        if not documents:
            if force_answer:
                print(f"Forcing answer without context.")
            else:
                return "No relevant documents found."

        context = '\n\n'.join(documents or [])
        prompt_with_context = f"{generator_preprompt}\n\n*Context:*\n{context}\n\n*Question:*\n{question}" if documents else question
        debug and print(f"Question with context: {prompt_with_context}")

        context_info_text = self._get_context_info_text(
            metadatas, distances, context_detail_top_k, debug).strip() if context_details and context else None

        len_docs = len(documents) if documents else 0

        if stream:
            return self._prompt_llm_stream(
                len_docs, prompt_with_context, context_info_text, gen_max_tokens, gen_temperature)
        else:
            return self._prompt_llm(
                len_docs, prompt_with_context, context_info_text, gen_max_tokens, gen_temperature)

    def _prompt_llm(
        self,
        n_retrieved: int,
        prompt_with_context: str,
        context_info_text: str | None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        llm_answer = self.llm.generate(
            prompt_with_context, max_tokens, temperature, stream=False)

        if context_info_text:
            llm_answer += f"\n\nContext information (RAG):\n{context_info_text}"

        return f"Retrieved {n_retrieved} documents.\n\n{llm_answer}"

    def _prompt_llm_stream(
        self,
        n_retrieved: int,
        prompt_with_context: str,
        context_info_text: str | None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        yield f"Retrieved {n_retrieved} documents.\n\n"

        for token in self.llm.generate(
            prompt_with_context, max_tokens, temperature, stream=True
        ):
            yield token

        if context_info_text:
            yield f"\n\nContext information (RAG):\n{context_info_text}"

    def _get_context_info_text(
        self,
        metadatas: List[dict],
        distances: List[float],
        context_detail_top_k: int | None = None,
        debug: bool = False,
    ):
        context_info = []
        for meta, dist in zip(metadatas, distances):
            unique_video_name = meta.get('unique_video_name', '')
            if not unique_video_name:
                course = 'Unknown course'
                video_name = 'Unknown video'
            else:
                videos_folder_path_index = unique_video_name.split(
                    '/').index('videos') if 'videos' in unique_video_name.split('/') else None

                if videos_folder_path_index == None:
                    debug and print(
                        f"'videos' not found in unique_video_name. unique_video_name: {unique_video_name}. Assuming that the first folder is the course and the second folder is the video.")
                    videos_folder_path_index = -1

                course = unique_video_name.split(
                    '/')[videos_folder_path_index + 1] if '/' in unique_video_name else 'Unknown course'
                video_name = unique_video_name.split(
                    '/')[videos_folder_path_index + 2] if '/' in unique_video_name else unique_video_name

            modality = meta.get('source', '').upper(
            )
            start = meta.get('start', 0.0)
            end = meta.get('end', 0.0)
            duration = end - start
            timestamp = (f"{int(start//3600)}h {int((start%3600)//60)}min {int(start%60)}s - "
                         f"{int(end//3600)}h {int((end%3600)//60)}min {int(end%60)}s "
                         f"({int(duration//60)}min {int(duration%60)}s)")
            distance = f"{dist:.2f}"
            context_info.append({
                'course': course,
                'video': video_name,
                'modality': modality,
                'timestamp': timestamp,
                'distance': distance
            })

        if context_detail_top_k:
            context_info = context_info[:context_detail_top_k]

        grouped_context = {}
        for info in context_info:
            course = info['course']
            video = info['video']
            if course not in grouped_context:
                grouped_context[course] = {}
            if video not in grouped_context[course]:
                grouped_context[course][video] = []
            grouped_context[course][video].append(info)

        context_info_text = ""
        for course, videos in grouped_context.items():
            context_info_text += f"Course: {course}\n"
            for video, infos in videos.items():
                context_info_text += f"  Video: {video}\n"
                for i, info in enumerate(infos):
                    context_info_text += (
                        f"    Sequence {i + 1} ({info['modality']}, {info['distance']}): {info['timestamp']}\n"
                    )
                context_info_text += "\n"

        return context_info_text
