import os
import os.path as osp
import argparse
import pickle
import matplotlib.pyplot as plt


def plot_probing_results(data_dir):
    plt.style.use('seaborn-v0_8-darkgrid')
    COLORS = ['#E69F00', '#0072B2']
    # plt.style.use('bmh')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'

    models = ["pixel-bigrams", "mpixel"]
    model_labels = {
        "pixel-base-bigrams": "PIXEL-BIGRAMS",
        "pixel-m4": "PIXEL-M4",
    }
    model_markers = {
        "mpixel": "o",          # circle for mPIXEL
        "pixel-bigrams": "s"    # square for PIXEL
    }

    languages = ["arabic", "armenian", "modern-greek", "macedonian", "russian"]
    lang_titles = {
        lang: ("Greek" if lang == "modern-greek" else lang.capitalize())
        for lang in languages
    }

    # tasks = ["Case", "POS", "SameFeat", "TagCount"]
    tasks = ["Case", "SameFeat"]

    # create grid, sharing y-axis across all plots
    fig, axes = plt.subplots(
        nrows=len(tasks),
        ncols=len(languages),
        figsize=(4 * len(languages), 3 * len(tasks)),
        sharey="row",
        sharex="col",
    )

    for i, task in enumerate(tasks):
        for j, lang in enumerate(languages):
            ax = axes[i,j]

            for model in models:
                fname = os.path.join(data_dir, f"{model}-{lang}-{task}-layer_all.pickle")
                if not os.path.isfile(fname):
                    print(f"Warning: {fname} not found.")
                    continue

                with open(fname, "rb") as f:
                    data = pickle.load(f)

                layers = sorted(data.keys())
                test_accs = [data[layer][1] for layer in layers]
                color = COLORS[0] if model == 'pixel-bigrams' else COLORS[1] 
                ax.plot(
                    layers,
                    test_accs,
                    label=model_labels[model],
                    marker=model_markers[model],
                    linestyle='-',
                    color=color,
                )

            if i == 0:
                ax.set_title(lang_titles[lang], pad=10, fontsize=24)
            if j == 0:
                ax.set_ylabel(task, rotation=90, labelpad=20, va='center', fontsize=20)

            if i == len(tasks) - 1:
                ax.set_xticks(range(1, 13))

    # put legend inside the top-left subplot
    ax0 = axes[-1,0]
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(
        handles,
        labels,
        loc='upper left',
        frameon=True,
        fontsize=14,
    )

    fig.tight_layout()
    fig.supxlabel('Layer', fontsize=20)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--results-dir", default="outputs/analysis/retrieval", type=str)
    args = parser.parse_args()
    plot_probing_results(osp.abspath(osp.expanduser(args.results_dir)))

