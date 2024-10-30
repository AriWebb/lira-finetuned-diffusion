from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from torch import autocast
from PIL import Image
import torchvision.transforms as transforms
import model_confidence

# Number of diffusion steps
# Also known as T later on
NUM_INFERENCE_STEPS = 50

SIGMA_STEPS_CAP = 1000

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# Use DDIMScheduler
pipe.scheduler = DDIMScheduler.from_config(
  pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing", num_train_timesteps=100
)

img = Image.open("../datasets/MOODENG/IMG_3786.jpg")

preprocess = transforms.Compose([
  transforms.Resize((512, 512)),
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5]) # Stable Diffusion normalization
])

# Generate z_0
img_tensor = preprocess(img).unsqueeze(0).to("cuda")
# not sure if have to multiply by .18??
latents = pipe.vae.encode(img_tensor).latent_dist.sample() * 0.18215 
print(latents.shape)

# noise = torch.randn(latents.shape).to("cuda")
# guidance_scale = 7.5 # This is the classifier-free guidance scale

# Conditioning (prompt)
prompt = "a pygmy hippo"
text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")


with torch.no_grad():
  with autocast("cuda"):
    # Generate y for calling solutions
    text_embeddings = pipe.text_encoder(text_input)[0]

    # Sample calls to model_confidence for solution_1, solution_2
    # Make sure pipe has scheduler set up.
    # TODO: Workout SIGMA_STEPS_CAP vs NUM_INFERENCE_STEPS
    model_confidence.solution_1(pipe, latents, text_embeddings, SIGMA_STEPS_CAP)
    model_confidence.solution_2(pipe, latents, text_embeddings, SIGMA_STEPS_CAP, NUM_INFERENCE_STEPS)

    t_index = torch.tensor([1], device="cuda", dtype=torch.long)

    denoising_vector = pipe.unet(latents, t_index, encoder_hidden_states=text_embeddings).sample

    print(denoising_vector.shape)
