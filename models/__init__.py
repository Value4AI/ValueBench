# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tqdm import tqdm

from .models import *

# A dictionary mapping of model architecture to its supported model names
MODEL_LIST = {
    T5Model: ['google/flan-t5-large'],
    LlamaAPIModel: [ # We use LlamaAPI for these models, one can also implement them locally
        'llama-7b-chat',
        'llama-7b-32k',
        'llama-13b-chat',
        'llama-70b-chat',
        'mixtral-8x7b-instruct',
        'mistral-7b-instruct',
        'mistral-7b',
        'NousResearch/Nous-Hermes-Llama2-13b',
        'falcon-7b-instruct',
        'falcon-40b-instruct',
        'alpaca-7b',
        'codellama-7b-instruct',
        'codellama-13b-instruct',
        'codellama-34b-instruct',
        'openassistant-llama2-70b',
        'vicuna-7b',
        'vicuna-13b',
        'vicuna-13b-16k',
    ],
    LlamaModel: ['llama2-7b', 'llama2-7b-chat', 'llama2-13b', 'llama2-13b-chat', 'llama2-70b', 'llama2-70b-chat',],
    PhiModel: ['phi-1.5', 'phi-2'],
    PaLMModel: ['palm'],
    OpenAIModel: ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o'],
    VicunaModel: ['vicuna-7b', 'vicuna-13b', 'vicuna-13b-v1.3'],
    UL2Model: ['google/flan-ul2'],
    GeminiModel: ['gemini-pro'],
    MistralModel: ['mistralai/Mistral-7B-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1'],
    MixtralModel: ['mistralai/Mixtral-8x7B-v0.1'],
    YiModel: ['01-ai/Yi-6B', '01-ai/Yi-34B', '01-ai/Yi-6B-Chat', '01-ai/Yi-34B-Chat'],
    BaichuanModel: ['baichuan-inc/Baichuan2-7B-Base', 'baichuan-inc/Baichuan2-13B-Base',
                    'baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-13B-Chat'],
}

SUPPORTED_MODELS = [model for model_class in MODEL_LIST.keys() for model in MODEL_LIST[model_class]]


class LLMModel(object):
    """
    A class providing an interface for various language models.

    This class supports creating and interfacing with different language models, handling prompt engineering, and performing model inference.

    Parameters:
    -----------
    model : str
        The name of the model to be used.
    max_new_tokens : int, optional
        The maximum number of new tokens to be generated (default is 20).
    temperature : float, optional
        The temperature for text generation (default is 0).
    device : str, optional
        The device to be used for inference (default is "cuda").
    dtype : str, optional
        The loaded data type of the language model (default is "auto").
    model_dir : str or None, optional
        The directory containing the model files (default is None).
    system_prompt : str or None, optional
        The system prompt to be used (default is None).
    api_key : str or None, optional
        The API key for API-based models (GPT series and Gemini series), if required (default is None).

    Methods:
    --------
    _create_model(max_new_tokens, temperature, device, dtype, model_dir, system_prompt, api_key)
        Creates and returns the appropriate model instance.
    convert_text_to_prompt(text, role)
        Constructs a prompt based on the text and role.
    concat_prompts(prompt_list)
        Concatenates multiple prompts into a single prompt.
    _gpt_concat_prompts(prompt_list)
        Concatenates prompts for GPT models.
    _other_concat_prompts(prompt_list)
        Concatenates prompts for non-GPT models.
    __call__(input_text, **kwargs)
        Makes a prediction based on the input text using the loaded model.
    """
    
    @staticmethod
    def model_list():
        return SUPPORTED_MODELS

    def __init__(self, model, max_new_tokens=20, temperature=0, device="cuda", dtype="auto", model_dir=None, system_prompt=None, api_key=None):
        self.model_name = model
        self.model = self._create_model(max_new_tokens, temperature, device, dtype, model_dir, system_prompt, api_key)

    def _create_model(self, max_new_tokens, temperature, device, dtype, model_dir, system_prompt, api_key):
        """Creates and returns the appropriate model based on the model name."""

        # Dictionary mapping of model names to their respective classes
        model_mapping = {model: model_class for model_class in MODEL_LIST.keys() for model in MODEL_LIST[model_class]}

        # Get the model class based on the model name and instantiate it
        model_class = model_mapping.get(self.model_name)
        if model_class:
            if model_class == LlamaAPIModel:
                return model_class(self.model_name, max_new_tokens, temperature, system_prompt, api_key)
            elif model_class == LlamaModel:
                return model_class(self.model_name, max_new_tokens, temperature, device, dtype, system_prompt, model_dir)
            elif model_class == VicunaModel:
                return model_class(self.model_name, max_new_tokens, temperature, device, dtype, model_dir)
            elif model_class in [OpenAIModel]:
                return model_class(self.model_name, max_new_tokens, temperature, system_prompt, api_key)
            elif model_class in [PaLMModel, GeminiModel]:
                return model_class(self.model_name, max_new_tokens, temperature, api_key)
            else:
                return model_class(self.model_name, max_new_tokens, temperature, device, dtype)
        else:
            raise ValueError("The model is not supported!")
    
    def __call__(self, input_texts, **kwargs):
        """Predicts the output based on the given input text using the loaded model."""
        if isinstance(self.model, OpenAIModel) or isinstance(self.model, LlamaAPIModel):
            return self.model.batch_predict(input_texts, **kwargs)
        else:
            responses = []
            for input_text in tqdm(input_texts):
                responses.append(self.model.predict(input_text, **kwargs))
            return responses
