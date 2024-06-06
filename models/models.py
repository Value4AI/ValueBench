# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from abc import ABC
import concurrent.futures
from tqdm import tqdm
import time

from openai import OpenAI
try:
    import torch
except ImportError:
    print("PyTorch is not installed. Using API models only.")


class LLMBaseModel(ABC):
    """
    Abstract base class for language model interfaces.

    This class provides a common interface for various language models and includes methods for prediction.

    Parameters:
    -----------
    model : str
        The name of the language model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    __call__(input_text, **kwargs)
        Shortcut for predict method.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device='auto'):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature,
                                     do_sample=True,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0])
        return out

    def __call__(self, input_text, **kwargs):
        return self.predict(input_text, **kwargs)


class BaichuanModel(LLMBaseModel):
    """
    Language model class for the Baichuan model.

    Inherits from LLMBaseModel and sets up the Baichuan language model for use.

    Parameters:
    -----------
    model : str
        The name of the Baichuan model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(BaichuanModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True)


class YiModel(LLMBaseModel):
    """
    Language model class for the Yi model.

    Inherits from LLMBaseModel and sets up the Yi language model for use.

    Parameters:
    -----------
    model : str
        The name of the Yi model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(YiModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)


class MixtralModel(LLMBaseModel):
    """
    Language model class for the Mixtral model.

    Inherits from LLMBaseModel and sets up the Mixtral language model for use.

    Parameters:
    -----------
    model : str
        The name of the Mixtral model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(MixtralModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)


class MistralModel(LLMBaseModel):
    """
    Language model class for the Mistral model.

    Inherits from LLMBaseModel and sets up the Mistral language model for use.

    Parameters:
    -----------
    model : str
        The name of the Mistral model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(MistralModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)


class PhiModel(LLMBaseModel):
    """
    Language model class for the Phi model.

    Inherits from LLMBaseModel and sets up the Phi language model for use.

    Parameters:
    -----------
    model : str
        The name of the Phi model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(PhiModel, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model = "microsoft/phi-1_5" if model_name == "phi-1.5" else "microsoft/phi-2"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, torch_dtype=dtype, device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=dtype, device_map=device)

    
    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0])
        return out[len(input_text):]

class T5Model(LLMBaseModel):
    """
    Language model class for the T5 model.

    Inherits from LLMBaseModel and sets up the T5 language model for use.

    Parameters:
    -----------
    model : str
        The name of the T5 model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(T5Model, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)


class UL2Model(LLMBaseModel):
    """
    Language model class for the UL2 model.

    Inherits from LLMBaseModel and sets up the UL2 language model for use.

    Parameters:
    -----------
    model : str
        The name of the UL2 model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype):
        super(UL2Model, self).__init__(model_name, max_new_tokens, temperature, device)
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=dtype, device_map=device)


class LlamaModel(LLMBaseModel):
    """
    Language model class for the Llama model.

    Inherits from LLMBaseModel and sets up the Llama language model for use.

    Parameters:
    -----------
    model : str
        The name of the Llama model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    system_prompt : str
        The system prompt to be used (default is 'You are a helpful assistant.').
    model_dir : str
        The directory containing the model files (default is None). If not provided, it will be downloaded from the HuggingFace model hub.
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype, system_prompt, model_dir):
        super(LlamaModel, self).__init__(model_name, max_new_tokens, temperature, device)
        if system_prompt is None:
            self.system_prompt = "You are a helpful assistant."
        else:
            self.system_prompt = system_prompt
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        if model_dir is None:
            parts = model_name.split('-')
            number = parts[1]
            is_chat = 'chat' in parts

            model_dir = f"meta-llama/Llama-2-{number}"
            if is_chat:
                model_dir += "-chat"
            model_dir += "-hf"
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map=device, torch_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, torch_dtype=dtype)

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device

        input_text = f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>\n{input_text}[/INST]"
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens, 
                                     temperature=self.temperature,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0], 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=False)
        
        return out[len(input_text):]


class VicunaModel(LLMBaseModel):
    """
    Language model class for the Vicuna model.

    Inherits from LLMBaseModel and sets up the Vicuna language model for use.

    Parameters:
    -----------
    model : str
        The name of the Vicuna model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    device: str
        The device to use for inference (default is 'auto').
    dtype: str
        The dtype to use for inference (default is 'auto').
    model_dir : str, optional
        The directory containing the model files (default is None).
    """
    def __init__(self, model_name, max_new_tokens, temperature, device, dtype, model_dir):
        super(VicunaModel, self).__init__(model_name, max_new_tokens, temperature, device)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map=device, torch_dtype=dtype, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, torch_dtype=dtype)

    def predict(self, input_text, **kwargs):
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        outputs = self.model.generate(input_ids, 
                                     max_new_tokens=self.max_new_tokens,
                                     temperature=self.temperature,
                                     **kwargs)
        
        out = self.tokenizer.decode(outputs[0])
        
        return out[len(input_text):]


class OpenAIModel(LLMBaseModel):
    """
    Language model class for interfacing with OpenAI's GPT models or Llama API models.

    Inherits from LLMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is 'You are a helpful assistant.').
    openai_key : str
        The OpenAI API key (default is None).

    Methods:
    --------
    predict(input_text)
        Predicts the output based on the given input text using the OpenAI model.
    """
    def __init__(self, model_name, max_new_tokens, temperature, system_prompt=None, openai_key=None):
        super(OpenAIModel, self).__init__(model_name, max_new_tokens, temperature)    
        self.openai_key = openai_key
        self.system_prompt = system_prompt

    def predict(self, input_text, kwargs={}):
        client = OpenAI(api_key=self.openai_key if self.openai_key is not None else os.environ['OPENAI_API_KEY'])
        if self.system_prompt is None:
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        else:
            system_messages = {'role': "system", 'content': self.system_prompt}
        
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]
        
        messages.insert(0, system_messages)
    
        # extra parameterss
        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        response_format = kwargs['response_format'] if 'response_format' in kwargs else None
        
        for attempt in range(1000):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    n=n,
                    response_format={"type": "json_object"} if response_format=="json" else None,
                )
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Retrying ({attempt + 1})...")
                time.sleep(1)
            
        if n > 1:
            result = [choice.message.content for choice in response.choices]
        else:
            result = response.choices[0].message.content
            
        return result

    def multi_predict(self, input_texts, **kwargs):
        """
        An example of input_texts:
        input_texts = ["Hello!", "How are you?", "Tell me a joke."]
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            args = [(messages, kwargs) for messages in input_texts]
            contents = executor.map(lambda p: self.predict(*p), args)
        return list(contents)

    def batch_predict(self, input_texts, **kwargs):
        assert "n" not in kwargs or kwargs["n"] == 1, "n > 1 is not supported for batch prediction."
        responses_list = []
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 200
        for start_idx in tqdm(range(0, len(input_texts), batch_size)):
            end_idx = min(start_idx + batch_size, len(input_texts))
            batch_input_texts = input_texts[start_idx: end_idx]
            batch_results_list = self.multi_predict(batch_input_texts, **kwargs)
            responses_list.extend(batch_results_list)
            # Save responses to file
            with open(f"temp-file-responses-{self.model_name}.txt", "a") as f:
                for response in batch_results_list:
                    f.write(response + "\n")
        return responses_list


class LlamaAPIModel(OpenAIModel):
    def __init__(self, model_name, max_new_tokens, temperature, system_prompt=None, llama_key=None):
        super(LlamaAPIModel, self).__init__(model_name, max_new_tokens, temperature, system_prompt, llama_key)
        self.system_prompt = system_prompt
        self.llama_key = llama_key
    
    def predict(self, input_text, kwargs={}):
        client = OpenAI(
                    api_key = self.llama_key if self.llama_key is not None else os.environ['LLAMA_API_KEY'],
                    base_url = "https://api.llama-api.com"
                    )
        if self.system_prompt is None:
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        else:
            system_messages = {'role': "system", 'content': self.system_prompt}
        
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]
        
        messages.insert(0, system_messages)
    
        # extra parameterss
        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        response_format = kwargs['response_format'] if 'response_format' in kwargs else None
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n,
            response_format={"type": "json_object"} if response_format=="json" else None,
        )
        
        if n > 1:
            result = [choice.message.content for choice in response.choices]
        else:
            result = response.choices[0].message.content
            
        return result


class PaLMModel(LLMBaseModel):
    """
    Language model class for interfacing with PaLM models.

    Inherits from LLMBaseModel and sets up a model interface for PaLM models.

    Parameters:
    -----------
    model : str
        The name of the PaLM model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    api_key : str, optional
        The PaLM API key (default is None).
    """
    def __init__(self, model, max_new_tokens, temperature=0, api_key=None):
        super(PaLMModel, self).__init__(model, max_new_tokens, temperature)
        self.api_key = api_key
    
    def predict(self, input_text, **kwargs):
        import google.generativeai as palm 
        
        palm.configure(api_key=self.api_key)
        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
        model = models[0].name
        
        n = kwargs['n'] if 'n' in kwargs else 1
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        
        completion = palm.generate_text(
            model=model,
            prompt=input_text,
            temperature=temperature,
            candidate_count = n,
            max_output_tokens=max_new_tokens,
        )
        
        if n > 1:
            result = [cand.output for cand in completion.candidates]
        else:
            result = completion.result
        
        return result
        
class GeminiModel(LLMBaseModel):
    """
    Language model class for interfacing with Google's Gemini models.

    Inherits from LLMBaseModel and sets up a model interface for Gemini models.

    Parameters:
    -----------
    model : str
        The name of the PaLM model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    gemini_key : str, optional
        The Gemini API key (default is None).
    """
    def __init__(self, model, max_new_tokens, temperature=0, gemini_key=None):
        super(GeminiModel, self).__init__(model, max_new_tokens, temperature)
        self.gemini_key = gemini_key
    
    def predict(self, input_text, **kwargs):
        import google.generativeai as genai 
        
        genai.configure(api_key=self.gemini_key)

        # Set up the model
        generation_config = {
        "temperature": self.temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": self.max_new_tokens,
        }

        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
        ]

        model = genai.GenerativeModel(model_name="gemini-pro",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)

        response = model.generate_content(input_text).text

        return response
    
if __name__ == "__main__":
    # Test LlamaAPIModel
    model_name = "llama-70b-chat"
    temperature = 0.
    max_new_tokens = 50

    model = LlamaAPIModel(model_name, max_new_tokens, temperature)
    input_texts = [
        "What is the weather like today?",
        "Hi?",
    ]
    responses = model.batch_predict(input_texts)
    print(responses)