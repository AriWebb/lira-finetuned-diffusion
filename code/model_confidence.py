from transformers import DeiTFeatureExtractor, DeiTModel
from diffusers import StableDiffusionPipeline
import torch

import hashlib
import io


def find_closest_timestep(s, alphas, alpha_lines):
  """
  Implements t(s) = argmin_t |alpha_line_s - alpha_t|
  Returns the timestep t where alpha_t is closest to alpha_line_s
  """
  # Calculate absolute differences and find minimum
  diffs = torch.abs(alphas - alpha_lines[s])
  return torch.argmin(diffs)

# Create cache before the loops
unet_cache = {}

def get_cached_unet(pipe: StableDiffusionPipeline, z, y, t, cache=unet_cache):
  # Create a cache key from the input tensors
  # Use detached tensors to avoid memory leaks
  cache_key = (z.detach().cpu().numpy().tobytes(), 
              y.detach().cpu().numpy().tobytes(), 
              t.item())  # Convert single tensor to scalar
  
  if cache_key not in cache:
    # If not in cache, compute and store
    with torch.no_grad():
        cache[cache_key] = pipe.unet(z, t, y).sample
  
  return cache[cache_key]

def get_alphas(pipe: StableDiffusionPipeline):
  # should be lengh NUM_INFERENCE_STEPS
  alphas = pipe.scheduler.alphas_cumprod.to("cuda")
  # alphas[0] = 1, so we are going to prepend
  alphas = torch.cat([torch.zeros(1, device="cuda"), alphas[:-1]])
  return alphas

#TODO: add typing and function definition so less confusing
def solution_1(pipe: StableDiffusionPipeline, z_0, y, K, sigma_steps_cap):
    torch.cuda.empty_cache()
    with torch.no_grad():
        alphas = get_alphas(pipe)
        Z = torch.zeros((sigma_steps_cap, *z_0.shape), device="cuda")
        # parallelized precomputation of alpha_lines
        alpha_lines = 1 / (torch.arange(sigma_steps_cap, device="cuda")**2 + 1)

        # Forward Diffusion
        for s in range(sigma_steps_cap-1):
            Z_line_cur = Z[s] / (torch.sqrt(alpha_lines[s]))
            t_s = find_closest_timestep(s, alphas, alpha_lines)
            if t_s == 0:
                # Use normal distribution sample when t_s is 0
                normal_sample = torch.randn_like(Z[s])
                Z_line_next = Z_line_cur + normal_sample
            else:
                # Regular case using UNet
                Z_line_next = Z_line_cur + get_cached_unet(pipe, Z[s], y, t_s)
            Z[s + 1] = alpha_lines[s+1] * Z_line_next

        # Reverse Diffusion
        Z_hat = torch.zeros_like(Z)
        for s in range(sigma_steps_cap-1, 0, -1):
            #print(s)
            t_s = find_closest_timestep(s, alphas, alpha_lines)
            z_hat_0 = (Z[s] - (torch.sqrt(1 - alpha_lines[s]) * get_cached_unet(pipe, Z[s], y, t_s))) / torch.sqrt(alpha_lines[s])
            if t_s == 0:
                Z_hat[s] = z_hat_0
            else:
                Z_hat[s] = torch.sqrt(alphas[t_s-1]) * z_hat_0 + torch.sqrt(1 - alphas[t_s-1]) * get_cached_unet(pipe, Z[s], y, t_s)

        # NOTE: parallelize this sum??
        print("FINISHED REVERSE, STARTING SUM")
        log_p = sum(
            torch.sum((Z_hat[s] - Z[s]) * (Z[s-1] - Z[s]))
            for s in range(1, len(Z_hat))
        )
        print("FINISHED SUM")
        global unet_cache
        unet_cache = {}
        base_prob = -0.5 * torch.norm(Z[sigma_steps_cap-1], p="fro") ** 2 
        return K * log_p + base_prob

#TODO: add typing and function definition so less confusing
def solution_2(pipe: StableDiffusionPipeline, z_0, y, N):
  # Z = torch.zeros((T, *z_0.shape), device="cuda")
  U = torch.zeros((N,), device="cuda")

  alphas = get_alphas(pipe)
  T = len(alphas)

  for i in range(N):
    #if i % 10 == 0:
      #print(f'on the {i} iteration')
    t = torch.randint(1, T-1, (1,), device="cuda")
    # Sample epsilon from standard normal distribution N(0, I)
    epsilon = torch.randn_like(alphas[t]).to("cuda")
    z_t = torch.sqrt(alphas[t]) * z_0 + torch.sqrt(1 - alphas[t]) * epsilon
    z_t_minus_one = torch.sqrt(alphas[t-1]) * z_0 + torch.sqrt(1 - alphas[t-1]) * epsilon
    z_hat_0 = (z_t - torch.sqrt(1 - alphas[t]) * pipe.unet(z_t, t, y).sample) / torch.sqrt(alphas[t])
    z_hat_t_minus_one = torch.sqrt(alphas[t-1]) * z_hat_0 + torch.sqrt(1 - alphas[t-1]) * pipe.unet(z_t, t, y).sample
    U[i-1] = torch.dot((z_hat_t_minus_one - z_t).view(-1), (z_t_minus_one - z_t).view(-1))
    # if i % 10 == 0:
    #   for i in [['z_t', z_t], ['z_t_minus_one',z_t_minus_one], ['z_hat_t_minus_one', z_hat_t_minus_one]]:
    #     print(f'{i[0]} sum: {torch.sum(i[1])}')  

  # Before calculating u_hat, check for NaN values in U
  if torch.isnan(U).any():
    print("Warning: U contains NaN values.")
    print("U:", U)

  u_hat = 1 / N * sum(U)
  return u_hat


# Adapted from https://github.com/py85252876/Reconstruction-based-Attack
def pang_solution(pipe: StableDiffusionPipeline, z_0, y, batch_num=10, inference=50):
    # Initialize DeiT model and feature extractor
    pipe.safety_checker = None
    feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-384")
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-384", add_pooling_layer=False)
    model.to("cuda")
    
    # Get embedding of target image (z_0)
    inputs_target = feature_extractor(z_0, return_tensors="pt")
    inputs_target = {key: value.to("cuda") for key, value in inputs_target.items()}
    with torch.no_grad():
        outputs_target = model(**inputs_target)
    target_embedding = outputs_target.last_hidden_state
    
    # Generate batch_num images and get their embeddings
    generated_embeddings = []
    for i in range(batch_num):
        # Generate image from prompt
        #print(y)
        image = pipe(y, num_inference_steps=inference, guidance_scale=7.5).images[0]
        image = image.convert("RGB")
        image.save(f"imgdump/{i}.png")
        # Get embedding
        inputs = feature_extractor(image, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        generated_embeddings.append(outputs.last_hidden_state)
    
    # Stack all embeddings
    generated_embeddings = torch.stack(generated_embeddings)
    
    # Compute similarity scores between target and generated images
    similarities = []
    for gen_embedding in generated_embeddings:
        # Using cosine similarity as default metric
        similarity = torch.nn.functional.cosine_similarity(
            target_embedding.squeeze(), 
            gen_embedding.squeeze(),
            dim=0
        ).mean()
        similarities.append(similarity.item())
    
    # Return average similarity score
    return sum(similarities) / len(similarities)

