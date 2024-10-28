from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from torch import autocast
from PIL import Image
import torchvision.transforms as transforms


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# Use DDIMScheduler
pipe.scheduler = DDIMScheduler.from_config(
  pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
)

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

# Draft forward diffusion code
def forward_diffusion(latents, num_steps):
    """
    Simulate forward diffusion by linearly adding noise to latents over t timesteps.
    
    Args:
    latents (torch.Tensor): Input latent tensor from VAE encoding
    num_steps (int): Number of diffusion steps
    
    Returns:
    List of torch.Tensor: Noisy latents at each timestep
    """
    device = latents.device
    noisy_latents = []
    
    for t in range(num_steps):
        # Generate noise with same shape as latents
        noise = torch.randn_like(latents).to(device)
        
        # Calculate the amount of noise to add (linear interpolation)
        alpha = t / (num_steps - 1)
        
        # Add noise to the latents
        noisy_latent = (1 - alpha) * latents + alpha * noise
        
        noisy_latents.append(noisy_latent)
    
    return noisy_latents

### Solution 1 ###

# Is this the right way to get T?
# Timesteps are in descending order (1000 -> 0)
timesteps = pipe.scheduler.timesteps

alphas = pipe.scheduler.alphas_cumprod[timesteps]

def sigma_step(alphas):
  return torch.sqrt((1 - alphas) / alphas)

sigmas = sigma_step(alphas)

# Assume we have z_0, y, T (ask how to get these)
Z = torch.zeros_like(sigmas)
# y is the prompt, I believe
y = prompt

# Not sure if this is the right way to get t
def find_closest_timestep(alphas_t, alpha_line_s, pipe):
  """
  Implements t(s) = argmin_t |alpha_line_s - alpha_t|
  Returns the timestep t where alpha_t is closest to alpha_line_s
  """
  # Calculate absolute differences and find minimum
  diffs = torch.abs(alphas_t - alpha_line_s)
  return torch.argmin(diffs)

# structure to keep track of z hat
Z_line = torch.zeros_like(Z)
# Forward Diffusion
# due to timesteps, sigmas decreases from index 0 -> 1000. is that desired?
for idx, s in enumerate(sigmas):
  alpha_line_s = 1 / (s**2 + 1)
  Z_line[idx] = Z[idx] / (torch.sqrt(alpha_line_s))
  t_s = find_closest_timestep(alphas, alpha_line_s, pipe)
  if t_s == 0:
    # Use normal distribution sample when t_s is 0
    normal_sample = torch.randn_like(Z[idx])
    Z_line[idx + 1] = Z_line[idx] + normal_sample
  else:
    # Regular case using UNet
    Z_line[idx + 1] = Z_line[idx] + pipe.unet(Z[idx], y, t_s)
  # NOTE: better way to precompute alpha_hat_s+1 
  Z[idx + 1] = Z_line[idx] / (torch.sqrt((s + 1)**2 + 1))

# Reverse Diffusion
Z_hat = torch.zeros_like(Z)
for idx, s in enumerate(torch.flip(sigmas)):
  # NOTE: reuse alpha_line from other for loop and t_s
  alpha_line_s = 1 / (s**2 + 1)
  t_s = find_closest_timestep(alphas, alpha_line_s, pipe)
  z_hat_0 = (Z[idx] - (torch.sqrt(1 - alpha_line_s) * pipe.unet(Z[idx], y, t_s))) / torch.sqrt(alpha_line_s)
  # NOTE: if t(s) = 0, set z_hat[idx] = z_hat_0
  Z_hat[idx] = pipe.unet(Z[idx], y, t_s) # two more terms in this I don't know how to do

log_p = sum(
  torch.sum((Z_hat[s] - Z[s]) * (Z[s-1] - Z[s]))
  for s in range(1, len(Z_hat))
)

# Solution 2

# TODO: Get number of trials (N) and (T) from somewhere
N = 10
T = 1000

# Assume we have z_0, y, T (ask how to get these)
Z = torch.zeros_like(sigmas)
# y is the prompt, I believe
y = prompt
z_0 = 0
U = torch.zeros(N, device="cuda")

for i in range(1, N+1):
  t = torch.randint(0, len(alphas), (1,), device="cuda")
  # Sample epsilon from standard normal distribution N(0, I)
  epsilon = torch.randn_like(Z[0]).to("cuda")
  z_t = torch.sqrt(alphas[t]) * z_0 + torch.sqrt(1 - alphas[t]) * epsilon
  z_t_minus_one = torch.sqrt(alphas[t-1]) * z_0 + torch.sqrt(1 - alphas[t-1]) * epsilon
  # Question: different T here for unet??
  z_hat_0 = (z_t - torch.sqrt(1 - alphas[t]) * pipe.unet(z_t, y, t))
  z_hat_t_minus_one = torch.sqrt(alphas[t-1]) * z_hat_0 + torch.sqrt(1 - alphas[t-1] * pipe.unet(z_t, y, t))
  U[i-1] = torch.dot((z_hat_t_minus_one - z_t), (z_t_minus_one - z_t))

u_hat = 1 / N * sum(U)
