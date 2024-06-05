import json
from RAG.storage import Storage
from typing import List, Literal, Any
from LLM.llama_cpp_model import LlamaCppLlm
from sentence_transformers import CrossEncoder

# ---

DEFAULT_DISTANCE_THRESHOLD = 1


class Retriever:
    def __init__(
        self,
        storage: Storage,
        llm: LlamaCppLlm | None = None,
        reranker: CrossEncoder | None = None
    ):
        self.storage = storage
        self.llm = llm
        self.reranker = reranker

    def get_context_data(
        self,
        question: str,
        n_docs: int = 5,
        modalities: List[Literal['asr', 'ocr', 'asr-ocr-avg']
                         ] = ['asr', 'ocr', 'asr-ocr-avg'],
        distance_threshold: int = DEFAULT_DISTANCE_THRESHOLD,
        context_preparation_scheme: Literal['add_text_descriptions',
                                            'llm_filter_relevant'] | None = None,
        reranker_n_docs: int = 30,
        debug: bool = False,
    ):
        if not modalities:
            raise ValueError("At least one source must be provided.")

        if context_preparation_scheme == 'llm_filter_relevant' and not self.llm:
            raise ValueError(
                "An LLM model must be provided for the 'llm_filter_relevant' context preparation scheme.")

        reranking = self.reranker is not None and reranker_n_docs is not None

        if reranking and n_docs > reranker_n_docs:
            reranker_n_docs = n_docs
            print(
                f"n_docs cannot be greater than reranker_n_docs. reranker_n_docs set to {n_docs}.")

        embedded_question = self.storage.embedder.embed_query(question)
        found = self.storage.collection.query(
            query_embeddings=embedded_question,
            where={"source": {"$in": modalities}},
            n_results=reranker_n_docs if reranking else n_docs,
            include=['distances', 'metadatas', 'documents']
        )

        if not found.get('documents')[0]:
            return None, None, None

        metadatas = []
        documents = []
        distances = []
        for meta, dist, doc in zip(found['metadatas'][0], found['distances'][0], found['documents'][0]):
            if dist <= distance_threshold:
                metadatas.append(meta)
                documents.append(doc)
                distances.append(dist)

        if not documents:
            dist_round = ', '.join([str(round(dist, 2))
                                    for dist in found['distances'][0]])
            print(
                f"All found documents are below the distance threshold of {distance_threshold} ({dist_round}).")
            return None, None, None

        if reranking:
            sentence_pairs = [[question, doc] for doc in documents]

            scores = self.reranker.predict(sentences=sentence_pairs)

            sorted_doc_indices = sorted(
                range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            top_k_indices = sorted_doc_indices[:n_docs]

            documents = [documents[idx] for idx in top_k_indices]
            metadatas = [metadatas[idx] for idx in top_k_indices]
            distances = [distances[idx] for idx in top_k_indices]

        if context_preparation_scheme == 'add_text_descriptions' and ('asr' in modalities or 'ocr' in modalities):
            documents = [f"{meta.get('source').upper()} text: {doc}"
                         for meta, doc in zip(metadatas, documents)]
        elif context_preparation_scheme == 'llm_filter_relevant':
            indexed_documents = [{'index': idx, 'document': doc}
                                 for idx, doc in enumerate(documents)]
            prompt = f"Given the following 'Question' and 'JSON Data', return a list of indexes with the documents that are relevant to the 'Question'.\n\nQuestion: {question}\n\nJSON Data:\n{json.dumps(indexed_documents)}\n\nYOU MUST NOT ANSWER ANYTHING ELSE! ANSWER ONLY WITH THE JSON LIST OF RELEVANT INDEXES!\n\nExample output: [0, 2, 3].\n\nExample output 2: [1, 3]\n\nOutput:"
            prompt_token_len = self._calculate_token_count(prompt)
            context_token_len = sum(self._calculate_token_count(doc)
                                    for doc in documents)
            safety_factor = 1.3
            max_tokens = int(
                (prompt_token_len + context_token_len) * safety_factor)
            temperature = 0.0
            answers = self.llm.generate(prompt, max_tokens, temperature)
            try:
                relevant_indexes = json.loads(answers)
                debug and print(
                    f"N documents before filtering: {len(documents)}, after filtering: {len(relevant_indexes)}.")
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

    def _calculate_token_count(self, text):
        # Assuming an average of 1.3 tokens per word (https://www.anyscale.com/blog/num-every-llm-developer-should-know)
        words = text.split()
        return int(len(words) * 1.3)
