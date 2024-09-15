import os
import wget

# ---


def inst_prompt_template(prompt: str, system_message: str | None):
    acc_prompt = '\n\n'.join(filter(bool, [system_message, prompt]))
    return f"<s>[INST] {acc_prompt} [/INST]"


def chatml_prompt_template(prompt: str, system_message: str | None):
    acc_prompt = f'<|im_start|>system\n{system_message}<|im_end|>' if system_message else ''
    acc_prompt += f'<|im_start|>user\n{prompt}<|im_end|>'
    return f"{acc_prompt}\n<|im_start|>assistant"


def phi_2_prompt_template(prompt: str, system_message: str | None):
    acc_prompt = '\n\n'.join(filter(bool, [system_message, prompt]))
    return f"Instruct: {acc_prompt}\n\nOutput:"


def phi_2_dpo_prompt_template(prompt: str, system_message: str | None):
    acc_prompt = '\n\n'.join(filter(bool, [system_message, prompt]))
    return f"### Human: {acc_prompt}\n\### Assistant:"


def phi_3_prompt_template(prompt: str, system_message: str | None):
    acc_prompt = f'<|system|>\n{system_message}<|end|>' if system_message else ''
    acc_prompt += f'<|user|>\n{prompt}<|end|>'
    return f"{acc_prompt}\n<|assistant|>"


def llama_3_prompt_template(prompt: str, system_message: str | None):
    # https://github.com/meta-llama/llama3/issues/29
    """
    As defined in tokenizer_config.json:
    {% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}

    And applied in llama.cpp main exe for example:
    ./main -m ~/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf --color -n -2 -e -s 0 -p '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n' -ngl 99 --mirostat 2 -c 8192 -r '<|eot_id|>' --in-prefix '\n<|start_header_id|>user<|end_header_id|>\n\n' --in-suffix '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' -i
    """
    acc_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>\n"
    acc_prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\n"
    acc_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    return acc_prompt


MODEL_CONFIG = {
    'Mistral 7B Instruct v0.2 Q5_K_M': {
        'url': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf',
        'prompt_template': inst_prompt_template,
    },
    'Mistral 7B CapybaraHermes-2.5 Q4_K_M': {
        'url': 'https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF/resolve/main/capybarahermes-2.5-mistral-7b.Q4_K_M.gguf',
        'prompt_template': inst_prompt_template,
    },
    'Llama 3 8B Instruct Nous Q5_K_M': {
        'url': 'https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf',
        'prompt_template': llama_3_prompt_template,
    },
    'Mixtral 13B Laser Dolphin DPO Q6_K': {
        'url': 'https://huggingface.co/TheBloke/laser-dolphin-mixtral-2x7b-dpo-GGUF/resolve/main/laser-dolphin-mixtral-2x7b-dpo.Q6_K.gguf',
        'prompt_template': chatml_prompt_template,
    },
    'Mixtral 46B Nous Hermes 2 DPO Q3_K_M': {
        'url': 'https://huggingface.co/TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/resolve/main/nous-hermes-2-mixtral-8x7b-dpo.Q3_K_M.gguf',
        'prompt_template': chatml_prompt_template,
    },
    'Mixtral 46B Nous Hermes 2 DPO Q5_K_M': {
        'url': 'https://huggingface.co/TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/resolve/main/nous-hermes-2-mixtral-8x7b-dpo.Q5_K_M.gguf',
        'prompt_template': chatml_prompt_template,
    },
    'Mixtral 46B Nous Hermes 2 DPO Q6_K': {
        'url': 'https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/resolve/main/Nous-Hermes-2-Mixtral-8x7B-DPO.Q6_K.gguf',
        'prompt_template': chatml_prompt_template,
    },
    'Llama 3 70B Instruct Nous Q3_K_S': {
        'url': 'https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q3_K_S.gguf',
        'prompt_template': llama_3_prompt_template,
    },
    'Llama 3 70B Instruct Nous Q5_K_M': {
        'url': 'https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q5_K_M.gguf',
        'prompt_template': llama_3_prompt_template,
    },
}


class LlamaCppLlm:
    def __init__(
        self,
        data_path: str,
        model_name: str,
        system_message: str = '',
        random_seed: int = None,
    ):
        self.model_name = model_name
        self.model_config = MODEL_CONFIG.get(model_name)

        if not self.model_config:
            possible_model_names = '\n'.join(MODEL_CONFIG.keys())
            raise ValueError(
                f"A defined model_name is required! Possible values:\n{possible_model_names}")

        if not os.path.exists(data_path):
            print(f"Creating data path for LLM: {data_path}")
            os.makedirs(data_path)

        model_file_name = self.model_config['url'].split('/')[-1]
        model_path = os.path.join(data_path, model_file_name)

        if not os.path.exists(model_path):
            print(f"Downloading model {model_name}...")
            wget.download(self.model_config['url'], model_path)

        self.system_message = system_message

        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=32768,  # The max context/sequence length to use
            n_threads=8,  # Max CPU threads to use
            n_gpu_layers=-1,  # Max GPU layers
            verbose=False
        )

        if random_seed:
            Llama.set_seed(self.llm, seed=random_seed)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ):
        if not prompt:
            raise ValueError("Prompt required for text generation!")

        prompt = self.model_config['prompt_template'](
            prompt, self.system_message)

        if stream:
            return self._generate_stream(prompt, max_tokens, temperature)
        else:
            return self._generate(prompt, max_tokens, temperature)

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        output = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.llm._token_eos,  # Probably not needed
            stream=False
        )
        relevant_output = output['choices'][0]['text'].strip()
        return relevant_output

    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ):
        output = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.llm._token_eos,  # Probably not needed
            stream=True
        )
        for i, response in enumerate(output):
            relevant_output = response['choices'][0]['text']
            yield relevant_output.strip() if i == 0 else relevant_output
