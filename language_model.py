import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


STYLE_PROMPTS = {
    "friendly": (
        "Respond in a friendly, helpful and reassuring tone."
    ),
    "formal": (
        "Respond in a professional and formal tone."
    ),
    "concise": (
        "Respond very briefly and directly. Use no more than two sentences."
    ),
    "enthusiastic": (
        "Respond in an upbeat and enthusiastic tone while remaining useful."
    ),
}


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

        logger.info(
            "Language model loaded successfully."
        )

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

        prompt_length = (
            inputs["input_ids"].shape[-1]
        )

        generated_tokens = (
            outputs[0][prompt_length:]
        )

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
                    "Extract the SMS message that the user wants checked "
                    "for spam. Return only the SMS text. Do not classify "
                    "it. Do not explain your answer. Do not add quotation "
                    "marks. If the entire input is already the SMS message, "
                    "return it unchanged."
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

        extracted = (
            extracted
            .strip()
            .strip('"')
            .strip("'")
        )

        if not extracted:
            logger.warning(
                "Empty extraction returned. "
                "Using original input."
            )

            return user_input.strip()

        logger.info(
            "Extracted SMS: %s",
            extracted,
        )

        return extracted

    def generate_response(
        self,
        original_input,
        extracted_sms,
        classification,
        confidence,
        style="friendly",
        retrieved_context=None,
    ):
        style_instruction = STYLE_PROMPTS.get(
            style,
            STYLE_PROMPTS["friendly"],
        )

        if confidence is not None:
            confidence_text = (
                f"{confidence * 100:.2f}%"
            )
        else:
            confidence_text = (
                "not available"
            )

        if retrieved_context:
            context_sections = []

            for item in retrieved_context:
                context_sections.append(
                    f"Source: {item['source']}\n"
                    f"Information: {item['text']}"
                )

            context_text = "\n\n".join(
                context_sections
            )

        else:
            context_text = (
                "No additional knowledge "
                "was retrieved."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an SMS spam detection assistant. "
                    "A machine learning classifier has already analysed "
                    "the SMS. Treat its classification as authoritative. "
                    "Do not invent or change the confidence score. "
                    "Use the retrieved knowledge when it is relevant to "
                    "explain why the message may be suspicious or "
                    "legitimate. Do not claim that the retrieved "
                    "information proves the classification. "
                    "If the result is spam, provide practical safety "
                    "advice. If the result is ham, say that it appears "
                    "legitimate based on the classifier but do not give "
                    "an absolute safety guarantee. "
                    f"{style_instruction}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original user input:\n"
                    f"{original_input}\n\n"
                    f"Extracted SMS:\n"
                    f"{extracted_sms}\n\n"
                    f"Classifier result:\n"
                    f"{classification}\n\n"
                    f"Classifier confidence:\n"
                    f"{confidence_text}\n\n"
                    f"Retrieved knowledge:\n"
                    f"{context_text}\n\n"
                    f"Respond to the user using the classification "
                    f"and relevant retrieved knowledge."
                ),
            },
        ]

        response = self._generate(
            messages,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.7,
        )

        if response:
            return response

        return self._fallback_response(
            classification,
            confidence,
        )

    def _fallback_response(
        self,
        classification,
        confidence,
    ):
        if confidence is not None:
            confidence_text = (
                f"{confidence * 100:.2f}%"
            )
        else:
            confidence_text = (
                "unknown confidence"
            )

        if classification == "spam":
            return (
                f"This was classified as spam with "
                f"{confidence_text}. Be cautious about replying, "
                f"clicking links or sharing personal information."
            )

        return (
            f"This was classified as ham with "
            f"{confidence_text}. It appears legitimate based on "
            f"the model, although no automated classifier can "
            f"guarantee that a message is safe."
        )
