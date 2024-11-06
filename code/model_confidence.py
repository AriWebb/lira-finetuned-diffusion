from diffusers import StableDiffusionPipeline
import torch


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
    cache[cache_key] = pipe.unet(z, t, y).sample
  
  return cache[cache_key]

def get_alphas(pipe: StableDiffusionPipeline):
  # should be lengh NUM_INFERENCE_STEPS
  alphas = pipe.scheduler.alphas_cumprod.to("cuda")
  # alphas[0] = 1, so we are going to prepend
  alphas = torch.cat([torch.zeros(1, device="cuda"), alphas[:-1]])
  return alphas

#TODO: add typing and function definition so less confusing
def solution_1(pipe: StableDiffusionPipeline, z_0, y, sigma_steps_cap):
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
    print(s)
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
  return log_p

#TODO: add typing and function definition so less confusing
def solution_2(pipe: StableDiffusionPipeline, z_0, y, N, T):
  # Z = torch.zeros((T, *z_0.shape), device="cuda")
  U = torch.zeros((N,), device="cuda")

  alphas = get_alphas(pipe)  

  for i in range(N):
    if i % 10 == 0:
      print(f'on the {i} iteration')
    t = torch.randint(1, T-1, (1,), device="cuda")
    # Sample epsilon from standard normal distribution N(0, I)
    epsilon = torch.randn_like(alphas[t]).to("cuda")
    z_t = torch.sqrt(alphas[t]) * z_0 + torch.sqrt(1 - alphas[t]) * epsilon
    z_t_minus_one = torch.sqrt(alphas[t-1]) * z_0 + torch.sqrt(1 - alphas[t-1]) * epsilon
    z_hat_0 = (z_t - torch.sqrt(1 - alphas[t]) * pipe.unet(z_t, t, y).sample) / torch.sqrt(alphas[t])
    z_hat_t_minus_one = torch.sqrt(alphas[t-1]) * z_hat_0 + torch.sqrt(1 - alphas[t-1]) * pipe.unet(z_t, t, y).sample
    U[i-1] = torch.dot((z_hat_t_minus_one - z_t).view(-1), (z_t_minus_one - z_t).view(-1))
    if i % 10 == 0:
      for i in [['z_t', z_t], ['z_t_minus_one',z_t_minus_one], ['z_hat_t_minus_one', z_hat_t_minus_one]]:
        
        print(f'{i[0]} sum: {torch.sum(i[1])}')  

  # Before calculating u_hat, check for NaN values in U
  if torch.isnan(U).any():
    print("Warning: U contains NaN values.")
    print("U:", U)

  u_hat = 1 / N * sum(U)
  return u_hat