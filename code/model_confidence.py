import torch

def find_closest_timestep(s, alphas, alpha_lines):
  """
  Implements t(s) = argmin_t |alpha_line_s - alpha_t|
  Returns the timestep t where alpha_t is closest to alpha_line_s
  """
  # Calculate absolute differences and find minimum
  diffs = torch.abs(alphas[s] - alpha_lines[s])
  return torch.argmin(diffs)

# Create cache before the loops
unet_cache = {}

def get_cached_unet(pipe, z, y, t, cache=unet_cache):
  # Create a cache key from the input tensors
  # Use detached tensors to avoid memory leaks
  cache_key = (z.detach().cpu().numpy().tobytes(), 
              y.detach().cpu().numpy().tobytes(), 
              t.item())  # Convert single tensor to scalar
  
  if cache_key not in cache:
    # If not in cache, compute and store
    cache[cache_key] = pipe.unet(z, y, t)
  
  return cache[cache_key]

def get_alphas(pipe):
  # should be lengh NUM_INFERENCE_STEPS
  alphas = pipe.scheduler.alphas_cumprod
  # alphas[0] = 1, so we are going to prepend
  alphas = torch.cat([torch.zeros(1, device=alphas.device), alphas[:-1]])
  return alphas

#TODO: add typing and function definition so less confusing
def solution_1(pipe, z_0, y, T):
  alphas = get_alphas(pipe)
  Z = torch.zeros((T, *z_0.shape), device="cuda")
  # parallelized precomputation of alpha_lines
  alpha_lines = 1 / (torch.arange(T, device="cuda")**2 + 1)

  # Forward Diffusion
  # TODO: try to cache unet calls? probably going to be the same forward and backward
  for s in range(T-1):
    Z_line_cur = Z[s] / (torch.sqrt(alpha_lines[s]))
    t_s = find_closest_timestep(s, alphas, alpha_lines)
    if t_s == 0:
      # Use normal distribution sample when t_s is 0
      normal_sample = torch.randn_like(Z[s])
      Z_line_next = Z_line_cur + normal_sample
    else:
      # Regular case using UNet
      Z_line_next = Z_line_cur[s] + get_cached_unet(pipe, Z[s], y, t_s)
    # TODO: NO dep on Z_line_next???
    Z[s + 1] = Z_line_cur / (torch.sqrt((s + 1)**2 + 1))

  # Reverse Diffusion
  Z_hat = torch.zeros_like(Z)
  for s in range(T, 1, -1):
    t_s = find_closest_timestep(alphas, alpha_lines[s], pipe)
    z_hat_0 = (Z[s] - (torch.sqrt(1 - alphas[s]) * get_cached_unet(pipe, Z[s], y, t_s))) / torch.sqrt(alphas[s])
    if t_s == 0:
      Z_hat[s] = z_hat_0
    else:
      Z_hat[s] = torch.sqrt(alphas[s-1]) * z_hat_0 + torch.sqrt(1 - alphas[s-1]) * get_cached_unet(Z[s], y, t_s)

  # NOTE: parallelize this sum??
  log_p = sum(
    torch.sum((Z_hat[s] - Z[s]) * (Z[s-1] - Z[s]))
    for s in range(1, len(Z_hat))
  )
  return log_p

#TODO: add typing and function definition so less confusing
def solution_2(pipe, z_0, y, T, N):
  # Z = torch.zeros((T, *z_0.shape), device="cuda")
  U = torch.zeros_like((T, *z_0.shape), device="cuda")

  alphas = get_alphas(pipe)  

  for i in range(1, N+1):
    t = torch.randint(0, len(alphas), (1,), device="cuda")
    # Sample epsilon from standard normal distribution N(0, I)
    epsilon = torch.randn_like(Z[0]).to("cuda")
    z_t = torch.sqrt(alphas[t]) * z_0 + torch.sqrt(1 - alphas[t]) * epsilon
    z_t_minus_one = torch.sqrt(alphas[t-1]) * z_0 + torch.sqrt(1 - alphas[t-1]) * epsilon
    z_hat_0 = (z_t - torch.sqrt(1 - alphas[t]) * pipe.unet(z_t, y, t)) / torch.sqrt(alphas[t])
    z_hat_t_minus_one = torch.sqrt(alphas[t-1]) * z_hat_0 + torch.sqrt(1 - alphas[t-1] * pipe.unet(z_t, y, t))
    U[i-1] = torch.dot((z_hat_t_minus_one - z_t), (z_t_minus_one - z_t))

  u_hat = 1 / N * sum(U)
  return u_hat