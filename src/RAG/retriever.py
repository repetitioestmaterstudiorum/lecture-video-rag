from RAG.storage import Storage
from typing import List, Literal, Any
from LLM.llama_cpp_model import LlamaCppLlm
import json

# ---

DEFAULT_DISTANCE_THRESHOLD = 1


def calculate_token_count(text):
    # Assuming an average of 1.3 tokens per word (https://www.anyscale.com/blog/num-every-llm-developer-should-know)
    words = text.split()
    return int(len(words) * 1.3)


class Retriever:
    def __init__(self, storage: Storage):
        self.storage = storage

    def get_context_data(
        self,
        question: str,
        n_docs: int = 5,
        sources: List[Literal['asr', 'ocr', 'asr-ocr-avg']
                      ] = ['asr', 'ocr', 'asr-ocr-avg'],
        distance_threshold: int = DEFAULT_DISTANCE_THRESHOLD,
        context_preparation_scheme: Literal['add_text_descriptions',
                                            'llm_filter_relevant'] | None = None,
        llm: LlamaCppLlm | None = None,
        reranker: Any | None = None,
        reranker_n_docs: int = 30,
        debug: bool = False,
    ):
        if not sources:
            raise ValueError("At least one source must be provided.")

        if context_preparation_scheme == 'llm_filter_relevant' and not llm:
            raise ValueError(
                "An LLM model must be provided for the 'llm_filter_relevant' context preparation scheme.")

        reranking = reranker is not None and reranker_n_docs is not None

        embedded_question = self.storage.embedder.embed_query(question)
        found = self.storage.collection.query(
            query_embeddings=embedded_question,
            where={"source": {"$in": sources}},
            n_results=reranker_n_docs if reranking else n_docs,
            include=['distances', 'metadatas', 'documents']
        )

        if not found.get('documents')[0]:
            print(f"No documents were found for this query.")
            return None, None, None

        debug and print(f"{len(found['documents'][0])}", end='')
        metadatas = []
        documents = []
        distances = []
        for meta, dist, doc in zip(found['metadatas'][0], found['distances'][0], found['documents'][0]):
            if dist <= distance_threshold:
                metadatas.append(meta)
                documents.append(doc)
                distances.append(dist)
        debug and print(f'F{len(documents)}', end='')

        if not documents:
            dist_round = ', '.join([str(round(dist, 2))
                                    for dist in found['distances'][0]])
            print(
                f"All found documents are below the distance threshold of {distance_threshold} ({dist_round}).")
            return None, None, None

        if reranking:
            debug and print(f'-R', end='')
            sentence_pairs = [[question, doc] for doc in documents]

            scores = reranker.predict(sentences=sentence_pairs)

            sorted_doc_indices = sorted(
                range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            top_k_indices = sorted_doc_indices[:n_docs]

            documents = [documents[idx] for idx in top_k_indices]
            metadatas = [metadatas[idx] for idx in top_k_indices]
            distances = [distances[idx] for idx in top_k_indices]
            debug and print(f'/R', end='')

        debug and print('')

        if context_preparation_scheme == 'add_text_descriptions' and ('asr' in sources or 'ocr' in sources):
            documents = [f"{meta.get('source').upper()} text: {doc}"
                         for meta, doc in zip(metadatas, documents)]
        elif context_preparation_scheme == 'llm_filter_relevant':
            indexed_documents = [{'index': idx, 'document': doc}
                                 for idx, doc in enumerate(documents)]
            prompt = f"Given the following 'Question' and 'JSON Data', return a list of indexes with the documents that are relevant to the 'Question'.\n\nQuestion: {question}\n\nJSON Data:\n{json.dumps(indexed_documents)}\n\nYOU MUST NOT ANSWER ANYTHING ELSE! ANSWER ONLY WITH THE JSON LIST OF RELEVANT INDEXES!\n\nExample output: [0, 2, 3].\n\nExample output 2: [1, 3]\n\nOutput:"
            prompt_token_len = calculate_token_count(prompt)
            context_token_len = sum(calculate_token_count(doc)
                                    for doc in documents)
            safety_factor = 1.3
            max_tokens = int(
                (prompt_token_len + context_token_len) * safety_factor)
            temperature = 0.0
            answers = llm.generate(prompt, max_tokens, temperature)
            try:
                relevant_indexes = json.loads(answers)
                debug and print(f"Relevant indexes: {relevant_indexes}")
                documents = [documents[idx] for idx in relevant_indexes]
                metadatas = [metadatas[idx] for idx in relevant_indexes]
                distances = [distances[idx] for idx in relevant_indexes]
            except Exception as e:
                error = "Error while filtering relevant documents with the LLM. Leaving context unfiltered!"
                if debug:
                    error += f"\n{e}"
                print(f"{error}\n")
        elif context_preparation_scheme == 'add_same_timestamp_asr_ocr':
            for i, meta in enumerate(metadatas):
                found = self.storage.collection.get(
                    where={
                        '$and': [
                            {'source': {'$ne': meta['source']}},
                            {'unique_video_name': meta['unique_video_name']},
                            {'start': {'$gte': meta['start']}},
                            {'start': {'$lte': meta['end']}},
                        ]
                    },
                    include=['metadatas', 'documents']
                )
                if not found.get('documents'):
                    debug and print(
                        f"No additional documents found for {meta['unique_video_name']}.")
                    continue
                docs_with_meta = [{**meta, 'text': doc} for meta,
                                  doc in zip(found['metadatas'], found['documents'])]
                docs_with_meta = sorted(
                    docs_with_meta, key=lambda doc: doc['start'])
                additional_text = ''
                debug and print(
                    f"add_same_timestamp_asr_ocr is adding text from {len(docs_with_meta)} additional documents.")
                for doc in docs_with_meta:
                    additional_text = '\n'.join(
                        list(filter(bool, [additional_text, doc['text']])))
                if additional_text:
                    documents[i] += f"\n{additional_text}"

        return metadatas, distances, documents

    def ask(
        self,
        question: str,
        generation_llm: LlamaCppLlm,
        retriever_llm: LlamaCppLlm | None = None,
        n_docs: int = 5,
        sources: List[Literal['asr', 'ocr', 'asr-ocr-avg']
                      ] = ['asr', 'ocr', 'asr-ocr-avg'],
        distance_threshold: int = DEFAULT_DISTANCE_THRESHOLD,
        force_answer: bool = False,
        context_preparation_scheme: Literal['add_text_descriptions',
                                            'llm_filter_relevant'] | None = None,
        reranker: object | None = None,
        reranker_n_docs: int = 30,
        generation_preprompt: str = '',
        gen_max_tokens: int = 512,
        gen_temperature: float = 0.7,
        context_details: bool = True,
        context_detail_top_k: int | None = None,
        debug: bool = False
    ):
        if not context_detail_top_k:
            context_detail_top_k = n_docs
        if context_detail_top_k > n_docs:
            context_detail_top_k = n_docs
            print(
                f"Reranker n_docs cannot be greater than n_docs. Setting context_detail_top_k to {n_docs}.")

        metadatas, distances, documents = self.get_context_data(
            question=question,
            n_docs=n_docs,
            sources=sources,
            distance_threshold=distance_threshold,
            context_preparation_scheme=context_preparation_scheme,
            llm=retriever_llm,
            reranker=reranker,
            reranker_n_docs=reranker_n_docs,
            debug=debug
        )
        context_details and print(f"\nRetrieved {len(documents)} documents.")

        if debug and documents:
            for meta, dist, doc in zip(metadatas, distances, documents):
                print(
                    f"Dist: {dist:.2f}, source: {meta.get('source')}, doc: {doc[:50]}...")

        if not documents:
            if force_answer:
                print(f"Forcing answer without context.")
            else:
                return "No relevant documents found."

        rag_answer = self.prompt_llm(
            context=documents,
            question=question,
            llm=generation_llm,
            generation_preprompt=generation_preprompt,
            max_tokens=gen_max_tokens,
            temperature=gen_temperature,
            debug=debug
        )

        if context_details and documents:
            context_info_text = self.get_context_info_text(
                metadatas, distances, context_detail_top_k)

            return f"{rag_answer}\n\nContext information (RAG):\n{context_info_text.strip()}"
        else:
            return rag_answer

    def get_context_info_text(
        self,
        metadatas: List[dict],
        distances: List[float],
        context_detail_top_k: int | None = None
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
                    print(
                        f"  --Expected unique_video_name to contain 'videos' in the path. Found path: {unique_video_name}--  ")
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

    def prompt_llm(
        self,
        context: List[str] | None,
        question: str,
        llm: LlamaCppLlm,
        generation_preprompt: str = '',
        max_tokens: int = 512,
        temperature: float = 0.7,
        debug: bool = False
    ):
        context = '\n'.join(context or [])
        prompt_with_context = f"{generation_preprompt}\n\n*Context:*\n{context}\n\n*Question:*\n{question}" if context else question

        debug and print(f"Question with context: {prompt_with_context}")
        answer = llm.generate(prompt_with_context, max_tokens, temperature)

        return answer
