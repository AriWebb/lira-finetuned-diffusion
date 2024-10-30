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

# Don't need while we loop from 1 to SIGMA_STEPS_CAP
# def sigma_step(alphas):
#   return torch.sqrt((1 - alphas) / alphas)

# sigmas = sigma_step(alphas)