import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms
import requests
import os
import gradio as gr

from huggingface_hub import login
# login(token = '')
# os.environ['TRANSFORMERS_CACHE'] = '/home/ec2-user/.cache'

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model Device {device}")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"]
)

#torch.cuda.empty_cache()

model_name = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
# torch.nn.DataParallel(model).to(device)
processor = AutoProcessor.from_pretrained(model_name)

exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
bad_words_ids

def predict(image, input_1):
    desc = f"{input_1}"
    prompts = [
        "User:",
        image,
        desc,
        "Assistant:",
    ]
    inputs = processor(prompts, return_tensors="pt", debug=True).to(device)
    generate_ids = model.generate(**inputs, eos_token_id = exit_condition, bad_words_ids = bad_words_ids, max_length=1000)
    generate_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    generate_text
    return generate_text

input_1 = gr.Image(type='pil', label="Image for inference")
input_2 = gr.Textbox(value="Describe this image", label="Prompt")
inputs = [input_1, input_2]
outputs = "textbox"

title = "IDEFICS (Image-aware Decoder Enhanced a la Flamingo with Interleaved Cross-attentionS) Image to Text"

demo = gr.Interface(predict,
             inputs,
             outputs,
             title=title,
             theme="huggingface")
demo.launch(share=True)