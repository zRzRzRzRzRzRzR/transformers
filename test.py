import torch

from transformers import Glm4vForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("/mnt/GLM-4/glm-4v-9b-zr")
tokenizer = AutoTokenizer.from_pretrained("/mnt/GLM-4/glm-4v-9b-zr")

messages = [
    {
        "role": "user",
        "content": "Describe this image."
    }
]

# Preparation for inference
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image_inputs = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=[text],images=image_inputs,padding=True,return_tensors="pt")
breakpoint()
model = Glm4vForConditionalGeneration.from_pretrained(
    "/mnt/GLM-4/glm-4v-9b-zr", torch_dtype=torch.bfloat16, device_map="auto"
)


# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
