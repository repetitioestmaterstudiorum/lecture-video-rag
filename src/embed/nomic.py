from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
import numpy as np

# ---


class Embedder:
    """Use:
    embedder = Embedder()
    sentences = ['search_document: Some useful info',
                'search_query: What is useful?']
    embedder.embed_documents(sentences)

    Boilderplate code from https://huggingface.co/nomic-ai/nomic-embed-text-v1
    """

    def __init__(self, hf_model_name='nomic-ai/nomic-embed-text-v1', device='cpu'):
        if device == 'cuda':
            # TODO use device map (use all available GPUs in case available)
            #  This is a workaround to use 2 GPUs if more than 1 is available
            if torch.cuda.device_count() > 1:
                rng = np.random.default_rng()  # Avoids the random seed setting
                random_bit = rng.integers(0, 2)  # 0 or 1
                print(f"Embedding random bit: {random_bit}")
                device_embedder = f"cuda:{random_bit}"
            else:
                device_embedder = 'cuda:0'
        else:
            device_embedder = device

        #  use_fast explanation: https://github.com/huggingface/transformers/issues/5486?ref=assemblyai.com#issuecomment-1543040543
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=False)
        self.model = AutoModel.from_pretrained(
            hf_model_name, trust_remote_code=True)
        self.model.eval()
        self.model.to(device_embedder)
        self.device = device_embedder

    def embed_documents(self, sentences: list[str]):
        return self.__embed_sentences(sentences)

    def embed_query(self, sentence: str):
        return self.__embed_sentences([sentence])[0]

    def __embed_sentences(self, sentences: list[str]):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self.__mean_pooling(
            model_output, encoded_input['attention_mask'])

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
