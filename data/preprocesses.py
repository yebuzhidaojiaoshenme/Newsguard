import os
import json
import spacy
import torch
from transformers import BertTokenizer, BertModel
from PIL import Image
from torchvision import transforms

# 加载预训练的Spacy模型
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image)
    return image

def preprocess_data(text_dir, image_dir, caption_dir, output_file):
    data = []
    for text_file in os.listdir(text_dir):
        text_path = os.path.join(text_dir, text_file)
        with open(text_path, 'r') as f:
            text = f.read()

        image_path = os.path.join(image_dir, text_file.replace('.txt', '.jpg'))
        caption_path = os.path.join(caption_dir, text_file.replace('.txt', '.json'))

        if os.path.exists(image_path) and os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                caption = json.load(f)['caption']

            text_tokens = preprocess_text(text)
            image_tensor = preprocess_image(image_path)
            caption_tokens = preprocess_text(caption)

            data.append({
                'text': text_tokens,
                'image': image_tensor,
                'caption': caption_tokens
            })

    torch.save(data, output_file)


if __name__ == "__main__":
    preprocess_data('texts', 'images', 'captions', 'preprocessed_data.pt')
