pip install torch torchvision transformers pillow requests

from flask import Flask, request, send_file
from PIL import Image
import io
import requests
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

app = Flask(__name__)
hf_token = " "
# Initialize model with remote code trust
model = AutoModelForImageSegmentation.from_pretrained(
    'briaai/RMBG-2.0',
    trust_remote_code=True,token=hf_token
)
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

# Preprocessing transformations
image_size = (1024, 1024)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Get image URL from request
        image_url = request.json.get('image_url')
        response = requests.get(image_url)
        original_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(original_image).unsqueeze(0).to(model.device)
        
        # Inference
        with torch.no_grad():
            pred = model(input_tensor)[-1].sigmoid().cpu()
        
        # Post-process mask
        mask = transforms.ToPILImage()(pred.squeeze()).resize(original_image.size)
        
        # Apply alpha channel
        original_image.putalpha(mask)
        
        # Return result
        img_io = io.BytesIO()
        original_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
 
