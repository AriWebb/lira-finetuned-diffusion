import matplotlib.pyplot as plt
import seaborn as sns
import re

def plot_from_log():
  # Log the results to a file
  solution = "lira_2"
  N = 400
  K = 100
  with open('fprs.txt', 'r') as file:
    content = file.read()

  # Extract all floating-point numbers using regex
  numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)

  # Convert the strings to floats
  fprs = [float(num) for num in numbers]

  with open('tprs.txt', 'r') as file:
    content = file.read()

  # Extract all floating-point numbers using regex
  numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)

  # Convert the strings to floats
  tprs = [float(num) for num in numbers]
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

plot_from_log()