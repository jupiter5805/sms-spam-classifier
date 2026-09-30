import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


class TinyLlamaAssistant:
    def __init__(
        self,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ):
        self.model_name = model_name
        self.device = self._get_device()

        logger.info(
            "Loading language model %s on %s",
            self.model_name,
            self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info("Language model loaded successfully.")

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def _generate(
        self,
        messages,
        max_new_tokens=120,
        do_sample=False,
        temperature=None,
    ):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        generation_options = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample and temperature is not None:
            generation_options["temperature"] = temperature
            generation_options["top_p"] = 0.9

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_options,
            )

        prompt_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[0][prompt_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response.strip()

    def extract_sms(self, user_input):
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract an SMS message from a user's request. "
                    "Return only the SMS message that should be checked "
                    "for spam. Do not explain anything. Do not classify "
                    "the message. Do not add quotation marks. If the "
                    "entire user input is already an SMS message, return "
                    "it unchanged."
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]

        extracted = self._generate(
            messages,
            max_new_tokens=100,
            do_sample=False,
        )

        extracted = extracted.strip().strip('"').strip("'")

        if not extracted:
            logger.warning(
                "Language model returned empty extraction. "
                "Using original input."
            )

            return user_input.strip()

        logger.info(
            "Extracted SMS message: %s",
            extracted,
        )

        return extracted

    def generate_response(
        self,
        original_input,
        extracted_sms,
        classification,
        confidence,
    ):
        confidence_percentage = confidence * 100

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful SMS spam detection assistant. "
                    "A machine learning classifier has already analysed "
                    "the SMS. Treat its classification as authoritative. "
                    "Explain the result briefly and conversationally. "
                    "Do not change or second-guess the classification. "
                    "If the result is spam, advise the user to be "
                    "cautious. If it is ham, explain that it appears "
                    "legitimate, while avoiding absolute guarantees. "
                    "Keep the answer concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original user request: {original_input}\n"
                    f"Extracted SMS: {extracted_sms}\n"
                    f"Classifier result: {classification}\n"
                    f"Classifier confidence: "
                    f"{confidence_percentage:.2f}%"
                ),
            },
        ]

        response = self._generate(
            messages,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
        )

        if not response:
            return self._fallback_response(
                classification,
                confidence_percentage,
            )

        return response

    def _fallback_response(
        self,
        classification,
        confidence_percentage,
    ):
        if classification == "spam":
            return (
                f"This message was classified as spam with "
                f"{confidence_percentage:.2f}% confidence. "
                f"I'd recommend being cautious with links, "
                f"requests for money, or personal information."
            )

        return (
            f"This message was classified as legitimate with "
            f"{confidence_percentage:.2f}% confidence. "
            f"It does not appear to match the spam patterns "
            f"learned by the classifier."
        )
