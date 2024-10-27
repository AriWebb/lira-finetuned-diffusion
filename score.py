from diffusers import StableDiffusionPipeline
import torch
from torch import autocast
from PIL import Image
import torchvision.transforms as transforms


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

img = Image.open("../datasets/MOODENG/IMG_3786.jpg")

preprocess = transforms.Compose([

transforms.Resize((512, 512)),

transforms.ToTensor(),

transforms.Normalize([0.5], [0.5]) # Stable Diffusion normalization

])

img_tensor = preprocess(img).unsqueeze(0).to("cuda")

latents = pipe.vae.encode(img_tensor).latent_dist.sample() * 0.18215 

print(latents.shape)


noise = torch.randn(latents.shape).to("cuda")


# Number of diffusion steps

num_inference_steps = 50

guidance_scale = 7.5 # This is the classifier-free guidance scale


# Conditioning (prompt)
prompt = "a pygmy hippo"
text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")


with torch.no_grad():
  with autocast("cuda"):

    text_embeddings = pipe.text_encoder(text_input)[0]

    t_index = torch.tensor([1], device="cuda", dtype=torch.long)

    denoising_vector = pipe.unet(latents, t_index, encoder_hidden_states=text_embeddings).sample

    print(denoising_vector.shape)

def forward_diffusion(img_tensor, num_steps):
    """
    Simulate forward diffusion by linearly adding noise to an image over t timesteps.
    
    Args:
    img_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
    num_steps (int): Number of diffusion steps
    
    Returns:
    List of torch.Tensor: Noisy images at each timestep
    """
    device = img_tensor.device
    noisy_images = []
    
    for t in range(num_steps):
        # Generate noise
        noise = torch.randn_like(img_tensor).to(device)
        
        # Calculate the amount of noise to add (linear interpolation)
        alpha = t / (num_steps - 1)
        
        # Add noise to the image
        noisy_img = (1 - alpha) * img_tensor + alpha * noise
        
        noisy_images.append(noisy_img)
    
    return noisy_images
