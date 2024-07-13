import chromadb
from typing import Literal, List
from chromadb.config import Settings
from utils.hash import hash_256

# ---

# Cheatsheet: https://docs.trychroma.com/api-reference


class DB:
    def __init__(self, db_path: str | None):
        if db_path:
            self.client = chromadb.PersistentClient(
                path=db_path, settings=Settings(anonymized_telemetry=False))
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False))

    def delete_collection(self, name: str):
        try:
            self.client.delete_collection(name)
        except ValueError:
            print(f"Collection {name} does not exist")

    def get_collection(self, name: str, space: Literal['cosine', 'l2', 'ip'] = 'cosine'):
        collection_id = self.client.get_or_create_collection(
            name=name, metadata={'hnsw:space': space}).id
        collection = CustomCollection(
            client=self.client, 
            model={'id': collection_id, 'name': name}
        )

        return collection


class CustomCollection(chromadb.Collection):
    def __init__(self, client, model):
        super().__init__(client, model)
        self._model = model
    
    @property
    def id(self) -> str:
        return self._model['id']

    @property
    def name(self) -> str:
        return self._model['name']

    def insert(self, embeddings=None, metadatas=None, documents: List[str] | str = None):
        """
        Add data with IDs generated from SHA-256 hashes of the documents.
        This method inherits all functionality from the superclass's `add` method, but ignores images and uris.
        """
        if not documents:
            raise ValueError(
                f"The insert method expects at least one document")

        if isinstance(documents, str):
            docs, metas, embds = [documents], [metadatas], [embeddings]
        elif isinstance(documents, list):
            docs = documents
            metas = metadatas
            embds = embeddings
        else:
            raise ValueError(f"Unexpected documents type: {type(documents)}")

        ids = []
        for doc, meta in zip(docs, metas or [{} for _ in range(len(documents))]):
            original_hash = self.get_hash(
                meta.get('source', 'unknown') if meta else 'unknown', doc)
            ids.append(original_hash)

        final_ids = self.deduplicate_ids(ids)

        self.add(
            ids=final_ids,
            embeddings=embds,
            metadatas=metas,
            documents=docs,
        )

    def get_hash(self, source: str, document: str) -> str:
        hash_source = hash_256(source)
        hash_document = hash_256(document)
        combined_hash = f"{hash_source}-{hash_document}"
        return combined_hash

    def deduplicate_ids(self, ids):
        unique_ids = set()
        final_ids = []

        for id_ in ids:
            original_id = id_
            counter = 0
            while id_ in unique_ids or self.get_id_exists(id_):
                counter += 1
                id_ = f"{original_id}-{counter}"
            if original_id != id_:
                print(
                    f"Duplicate id found ({original_id}), added '-{counter}")
            unique_ids.add(id_)
            final_ids.append(id_)

        return final_ids

    def get_id_exists(self, id: str):
        doc_ids = self.get(ids=id).get('ids', [])
        return True if len(doc_ids) > 0 else False
