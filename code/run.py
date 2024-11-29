from typing import Optional
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch
from torch import autocast
from PIL import Image
import torchvision.transforms as transforms
import model_confidence
import os
import lira
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def map_index_eve(idx: int):
  if idx >= 64:
    idx += 1
  if idx >= 87:
    idx += 1
  if idx >= 128:
    idx += 1
  if idx >= 145:
    idx += 1
  return idx

def pad(idx: int):
  if idx < 10:
    return "00" + str(idx)
  elif idx < 100:
    return "0" + str(idx)
  else:
    return str(idx)

# Number of diffusion steps
# Also known as T later on
NUM_INFERENCE_STEPS = 50 # number of steps for the diffusion model
SIGMA_STEPS_CAP = 250 #1000 # arbitrary value to cap the number
NUMBER_TRIALS = 100 # maybe bump up to 1000 later, but takes a while to run solution_2

pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

# Use DDIMScheduler
pipe.scheduler = DDIMScheduler.from_config(
  pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing", num_train_timesteps=NUM_INFERENCE_STEPS
)

def generateLatent(filepath: str):
  # Load image with a relative path
  # image_path = os.path.join(os.path.dirname(__file__), f"../datasets/MOODENG/IMG_3786.jpg")
  image = Image.open(filepath)
  if not image.mode == "RGB":
    image = image.convert("RGB")

  img = np.array(image).astype(np.uint8)
  image = Image.fromarray(img)
  image = image.resize((512, 512), resample=Image.BICUBIC)
  image = transforms.RandomHorizontalFlip(p=0.5)(image)
  image = np.array(image).astype(np.uint8)
  image = (image / 127.5 - 1.0).astype(np.float32)
  x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to("cuda").half() 

  latents = pipe.vae.encode(x).latent_dist.sample().to("cuda") * pipe.vae.config.scaling_factor

  return latents

  """
  img = Image.open(filepath)

  preprocess = transforms.Compose([
      transforms.Resize((512, 512)),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]) # Stable Diffusion normalization
  ])

  # Generate z_0
  img_tensor = preprocess(img).unsqueeze(0).to("cuda")
  # not sure if have to multiply by .18??
  latents = pipe.vae.encode(img_tensor).latent_dist.sample() * 0.18215
  return latents
  """

def log_results(fprs, tprs, in_vals, out_vals, solution: str, K: Optional[float] = None, N: Optional[float] = None):
  # Log the results to a file
  with open("log2.txt", "a") as log_file:
    if solution == "lira_1":
      log_file.write(f"K={K}\n")
    else:
      log_file.write(f"N={N}\n")
    log_file.write(f"FPRs: {fprs}\n")
    log_file.write(f"TPRs: {tprs}\n")
    log_file.write(f"In Vals: {in_vals}\n")
    log_file.write(f"Out Vals: {out_vals}\n\n")

  plt.figure(figsize=(8, 6))

  sns.lineplot(x=fprs, y=tprs, label=f"Seed")

  # Finalize ROC plot
  plt.xlabel("FPR")
  plt.ylabel("TPR")
  if solution == "lira_1":
    plt.title(f"ROC Plot for K={K}")
  else: 
    plt.title(f"ROC Plot for N={N}")
  plt.plot([1e-4, 1], [1e-4, 1], color='lightgrey', linestyle='--', label="y=x")
  plt.legend()
  if solution == "lira_1":
    plt.savefig(f"../plots/roc_plot_lira_1_K={K}.png", format="png", dpi=300)
  else:
    plt.savefig(f"../plots/roc_plot_lira_2_N={N}.png", format="png", dpi=300)
  plt.close()

# noise = torch.randn(latents.shape).to("cuda")
# guidance_scale = 7.5 # This is the classifier-free guidance scale

# Conditioning (prompt)
prompt = "a painting"
text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to("cuda")
SHADOW_SEEDS = [1825, 410, 4507, 4013, 3658, 2287, 1680, 8936, 1425, 9675, 6913, 521, 489, 1536, 3583, 3812]

SPLITS = {
  "train" : [152, 6, 153, 52, 98, 78, 122, 2, 63, 20, 61, 74, 30, 129, 155, 57, 104, 14, 23, 150, 120, 73, 156, 85, 115, 51, 8, 105, 36, 108, 62, 112, 143, 25, 103, 16, 146, 124, 90, 107, 92, 75, 60, 114, 109, 127, 84, 123, 56, 157, 100, 141, 131, 145, 21, 139, 0, 46, 89, 95, 48, 80, 49, 125, 117, 64, 134, 77, 110, 128, 3, 11, 17, 140, 151, 81, 45, 26, 42, 10, 41, 5, 87, 149, 38, 72, 12, 28, 47, 22, 88, 133, 101, 4, 1, 44, 93, 40, 138, 97, 126, 113, 86, 96, 137, 68, 118, 33, 106, 147, 31, 29, 55, 18, 53, 111, 50, 142, 70],
  "in_raw" : [29, 25, 10, 1, 7, 2, 31, 26, 4, 17, 13, 11, 28, 6, 30, 5, 18, 35, 23, 3],
  "eval" : [43, 121, 82, 35, 15, 91, 135, 130, 34, 54, 116, 154, 37, 9, 99, 119, 71, 19, 32, 67, 13, 24, 76, 79, 83, 58, 65, 102, 59, 144, 66, 136, 132, 94, 39, 69, 7, 148, 27]
}
in_idxs = [SPLITS["eval"][i] for i in SPLITS["in_raw"]]
out_idxs = [i for i in range(158) if i not in (SPLITS["train"] + in_idxs)]

in_filepaths = [f"../../../datasets/evectrl/image-{pad(map_index_eve(idx))}.jpg" for idx in in_idxs]
out_filepaths = [f"../../../datasets/evectrl/image-{pad(map_index_eve(idx))}.jpg" for idx in out_idxs]
target_path = "../../../ti/eve_ctrl_target/64/learned_embeds-steps-10000.safetensors"
shadow_paths = [f"../../../ti/eve_ctrl_shadow/64/{shadow_seed}/learned_embeds-steps-10000.safetensors" for shadow_seed in SHADOW_SEEDS]
token = "<eve>"
granularity = 500 

with torch.no_grad():
    with autocast("cuda"):
    # Generate y for calling solutions
    #text_embeddings = pipe.text_encoder(text_input)[0]

    # Sample calls to model_confidence for solution_1, solution_2
    # Make sure pipe has scheduler set up.
    # TODO: Workout SIGMA_STEPS_CAP vs NUM_INFERENCE_STEPS
    # print(model_confidence.solution_1(pipe, latents, text_embeddings, SIGMA_STEPS_CAP))
    # print(model_confidence.solution_2(pipe, latents, text_embeddings, NUMBER_TRIALS, NUM_INFERENCE_STEPS))
        ins, outs = [], []
    # Maybe try to parellelize if this is a bottleneck
        for filepath in in_filepaths:
            ins.append(generateLatent(filepath))
    
        for filepath in out_filepaths:
            outs.append(generateLatent(filepath))

        for K in [0.0625, 0.25, 1, 4, 16, 64, 256]:
            fprs, tprs, in_vals, out_vals = lira.threshold_attack_1(pipe, prompt, SIGMA_STEPS_CAP, target_path, shadow_paths, ins, outs, token, granularity, K)
            t_index = torch.tensor([1], device="cuda", dtype=torch.long)

            #denoising_vector = pipe.unet(latents, t_index, encoder_hidden_states=text_embeddings).sample

            #print(denoising_vector.shape)
            log_results(fprs, tprs, in_vals, out_vals, "lira_1", K)
        
        for N in [10, 100, 200]:
            fprs, tprs, in_vals, out_vals = lira.threshold_attack_2(pipe, prompt, SIGMA_STEPS_CAP, target_path, shadow_paths, ins, outs, token, granularity, N)
            log_results(fprs, tprs, in_vals, out_vals, "lira_2", N)



