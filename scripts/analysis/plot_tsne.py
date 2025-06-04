import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup style and palette
plt.style.use('seaborn-v0_8-darkgrid')
# plt.style.use('bmh')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
#plt.style.use('bmh')
cmap = plt.get_cmap('tab20')
fixed_groups = ['Arabic', 'Brahmic', 'Chinese', 'Cyrillic', 'Devanagari', 'Japanese', 'Korean', 'Latin', 'Other']
color_map = {grp: cmap(i) for i, grp in enumerate(fixed_groups)}

# 2. Label grouping function
def map_group(label):
    if label.endswith('_Latn'):
        return 'Latin'
    if label.endswith('_Hans') or label.endswith('_Hant'):
        return 'Chinese'
    if label.endswith('_Deva'):
        return 'Devanagari'
    if label.endswith('_Cyrl'):
        return 'Cyrillic'
    if label.endswith('_Taml') or label.endswith('_Telu') or label.endswith('_Beng'):
        return 'Brahmic'
    if label.endswith('_Jpan'):
        return 'Japanese'
    if label.endswith('_Hang'):
        return 'Korean'
    if label.endswith('_Arab'):
        return 'Arabic'
    return 'Other'

# 3. Centroid definitions (merge Simplified/Traditional Chinese)
centroid_map = {
    'eng_Latn': ['eng_Latn'],
    'hin_Deva': ['hin_Deva'],
    'ukr_Cyrl': ['ukr_Cyrl'],
    'Chinese': ['zho_Hant', 'zho_Hans']
}

model_name_dict = {
    "pixel-base-bigrams": "PIXEL-BIGRAMS",
    "pixel-m4": "PIXEL-M4",
}

# 4. Parameters
models = ['pixel-base-bigrams', 'pixel-m4']
layers = [0, 4, 8, 12]
perp, lr, n_iter = 50, 500, 1000

# Pattern for file paths
output_pattern = "outputs/analysis/tsne/{model}-layer={layer}-n_components=2-perp={perp}-lr={lr}-n_iter={n_iter}.npz"

# 5. Gather global x/y extents
all_x, all_y = [], []
for model in models:
    for layer in layers:
        fn = output_pattern.format(model=model, layer=layer, perp=perp, lr=lr, n_iter=n_iter)
        if os.path.exists(fn):
            data = np.load(fn, allow_pickle=True)
            emb = data['embeddings']
            all_x.append(emb[:, 0])
            all_y.append(emb[:, 1])

if not all_x:
    raise FileNotFoundError("No matching .npz files found in outputs/")
all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
xmin, xmax = all_x.min(), all_x.max()
ymin, ymax = all_y.min(), all_y.max()

# 6. Create figure and axes
# fig, axes = plt.subplots(2, 4, figsize=(22, 10),
#                          gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
fig, axes = plt.subplots(
    2, len(layers), figsize=(22, 10),
    # gridspec_kw={'wspace': 0.1, 'hspace': 0.05},
    # constrained_layout=True
)

legend_handles, legend_labels = [], []

# 7. Plotting loop
for i, model in enumerate(models):
    for j, layer in enumerate(layers):
        ax = axes[i, j]
        fn = output_pattern.format(model=model, layer=layer, perp=perp, lr=lr, n_iter=n_iter)
        if not os.path.exists(fn):
            ax.set_title(f"{model.upper()}, Layer {layer}\n(no file)")
            ax.axis('off')
            continue
        
        data = np.load(fn, allow_pickle=True)
        emb = data['embeddings']
        lbls = data['labels']
        
        # Filter out unwanted labels
        mask = ~np.isin(lbls, ['taq_Latn', 'taq_Tfng'])
        emb, lbls = emb[mask], lbls[mask]
        
        # Scatter by script group
        groups = [map_group(lbl) for lbl in lbls]
        for grp in fixed_groups:
            grp_mask = np.array(groups) == grp
            if grp_mask.any():
                sc = ax.scatter(emb[grp_mask, 0], emb[grp_mask, 1],
                                alpha=0.6, color=color_map[grp], rasterized=True, zorder=1)
                if grp not in legend_labels:
                    legend_handles.append(sc)
                    legend_labels.append(grp)
        
        # Title and axis labels
        model_name = model_name_dict[model]
        if i == 0:
            ax.set_title(f"Layer {layer}", fontsize=20)
        if j == 0:
            ax.set_ylabel(model_name, fontsize=20, rotation=90, labelpad=10)
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # Enable ticks and set identical limits
        # Y‐ticks: only on leftmost column
        if j == 0:
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='y', which='both', labelleft=True, left=True)
        else:
            ax.tick_params(axis='y', which='both', labelleft=False, left=False)

        # X‐ticks: only on bottom row
        if i == len(models) - 1:
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='x', which='both', labelbottom=True, bottom=True)
        else:
            ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
       # ax.tick_params(which='both', labelsize=8)
        
        # Plot centroids for each defined cluster
        for src_labels in centroid_map.values():
            cmask = np.isin(data['labels'], src_labels)
            if cmask.any():
                center = data['embeddings'][cmask].mean(axis=0)
                ax.scatter(center[0], center[1], marker='x', s=100, linewidths=2, color='black')

# 8. Place legend next to the top-right subplot
# axes[0, 3].legend(legend_handles, legend_labels,
#                   loc='lower center', bbox_to_anchor=(1.0, 1.02),
#                   title='Script Groups', fontsize='medium', frameon=True)
if legend_labels:
    legend = fig.legend(legend_handles, legend_labels, loc='upper center', ncol=len(legend_labels), bbox_to_anchor=(0.5, 0.07), fontsize=24, markerscale=2.0,
        handlelength=0.6,       # default 2.0 → how long each little line is
        handletextpad=0.3,      # default 0.8 → gap between handle and text
        columnspacing=2.0,      # default 2.0 → gap between legend entries
        borderpad=0.2,          # default 0.4 → padding between contents and frame
        labelspacing=0.1        # default 0.5 → vertical spacing (doesn’t hurt even in 1-row))
    )



# 9. Final layout adjustments
# plt.subplots_adjust(hspace=0.1)
# plt.tight_layout(rect=[0, 0, 0.9, 1])
# plt.tight_layout(h_pad=0.05, w_pad=0.05)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()

