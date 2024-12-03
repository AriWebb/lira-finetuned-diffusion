import re
import matplotlib.pyplot as plt

# Initialize dictionaries to store data
data = {}

# Read and parse the log file
with open('log2.txt', 'r') as f:
    lines = f.readlines()
    current_k = None
    parsing_fprs = False
    parsing_tprs = False
    temp_fprs = []
    temp_tprs = []
    
    for line in lines:
        # Detect lines with K values
        if line.startswith("K="):
            if current_k is not None:
                # Store the accumulated FPR and TPR lists for the previous K value
                data[current_k] = {'FPRs': temp_fprs, 'TPRs': temp_tprs}
            # Reset for the new K value
            current_k = float(line.split('=')[1].strip())
            temp_fprs = []
            temp_tprs = []
            parsing_fprs = False
            parsing_tprs = False
        
        # Detect and start parsing FPRs
        elif line.startswith("FPRs:"):
            parsing_fprs = True
            parsing_tprs = False
            temp_fprs.extend(map(float, re.findall(r"[\d.]+", line)))
        
        # Detect and start parsing TPRs
        elif line.startswith("TPRs:"):
            parsing_tprs = True
            parsing_fprs = False
            temp_tprs.extend(map(float, re.findall(r"[\d.]+", line)))
        
        # Continue parsing if still in FPRs or TPRs
        elif parsing_fprs:
            temp_fprs.extend(map(float, re.findall(r"[\d.]+", line)))
            if line.strip().endswith(']'):  # End of FPRs list
                parsing_fprs = False
        
        elif parsing_tprs:
            temp_tprs.extend(map(float, re.findall(r"[\d.]+", line)))
            if line.strip().endswith(']'):  # End of TPRs list
                parsing_tprs = False

    # Store the final K value data after loop ends
    if current_k is not None:
        data[current_k] = {'FPRs': temp_fprs, 'TPRs': temp_tprs}

for k, values in data.items():
    fpr_tpr_pairs = {}
    for fpr, tpr in zip(values['FPRs'], values['TPRs']):
        if fpr in fpr_tpr_pairs:
            fpr_tpr_pairs[fpr] = max(fpr_tpr_pairs[fpr], tpr)  # Keep the maximum TPR for each FPR
        else:
            fpr_tpr_pairs[fpr] = tpr
    # Update the FPRs and TPRs lists with truncated data
    unique_fprs = sorted(fpr_tpr_pairs.keys())
    data[k]['FPRs'] = unique_fprs
    data[k]['TPRs'] = [fpr_tpr_pairs[fpr] for fpr in unique_fprs]

# Plot the ROC curves
plt.figure(figsize=(10, 8))
for k, values in data.items():
    plt.plot(values['FPRs'], values['TPRs'], label=f'K={k}')

# Plotting settings
plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line for random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Attack 1 ROC Curves for Different K Values')
plt.legend()
plt.grid()
plt.savefig(f'roc_curve_K1.png')
