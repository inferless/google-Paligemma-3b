from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch

class InferlessPythonModel:
    def initialize(self):
        model_id = "google/paligemma-3b-mix-224"
        device = "cuda:0"
        dtype = torch.bfloat16
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,
                                                                       torch_dtype=dtype,
                                                                       device_map=device,revision="bfloat16",
                                                                       token="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").eval()
        self.processor = AutoProcessor.from_pretrained(model_id,
                                                       token="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")

    def infer(self,inputs):
        prompt = inputs["prompt"]
        image_url = inputs["image_url"]
        image = Image.open(requests.get(image_url, stream=True).raw)
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        # Return a dictionary containing the result
        return {'response': decoded}

    def finalize(self):
        pass
