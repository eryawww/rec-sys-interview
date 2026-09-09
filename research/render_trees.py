"""Render every booster tree of the fitted watch_threshold model to PNG.

Writes decision-visualized/tree-001.png .. tree-100.png plus a contact
sheet. Each figure is one tree of the gradient-boosted ensemble: split
nodes as circles, leaves as squares, labelled down to LABEL_DEPTH.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from recsys.algorithms import build
from recsys.data import load_dataset

OUT = pathlib.Path("decision-visualized")
PAPER, INK, MUTED, ACCENT, SOFT = "#f5f5f5", "#2d3142", "#4f5d75", "#eb6c36", "#7a8399"
LABEL_DEPTH = 3


def feature_names(algo) -> list[str]:
    names = ["age"]
    names += [f"gender={g}" for g in algo._genders]
    names += [f"region={r}" for r in algo._regions]
    names += [f"type={c}" for c in algo._content_types]
    names += [f"genre={g}" for g in algo._genres]
    inverse = {v: k for k, v in algo._vectorizer.vocabulary_.items()}
    names += [f"title~{inverse[i]}" for i in range(len(inverse))]
    return names


def layout(nodes):
    """Return depth, x position and child links for every node."""
    depth = {}
    stack = [(0, 0)]
    while stack:
        idx, d = stack.pop()
        depth[idx] = d
        if not nodes[idx]["is_leaf"]:
            stack.append((int(nodes[idx]["left"]), d + 1))
            stack.append((int(nodes[idx]["right"]), d + 1))

    leaves: list[int] = []

    def collect(idx):
        if nodes[idx]["is_leaf"]:
            leaves.append(idx)
            return
        collect(int(nodes[idx]["left"]))
        collect(int(nodes[idx]["right"]))

    collect(0)
    slot = {n: k for k, n in enumerate(leaves)}

    x = {}

    def place(idx):
        if nodes[idx]["is_leaf"]:
            x[idx] = float(slot[idx])
            return x[idx]
        left = place(int(nodes[idx]["left"]))
        right = place(int(nodes[idx]["right"]))
        x[idx] = (left + right) / 2
        return x[idx]

    place(0)
    return depth, x, len(leaves)


def render(nodes, names, title, path):
    depth, x, n_leaves = layout(nodes)
    max_depth = max(depth.values())
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=110)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    for idx, node in enumerate(nodes):
        if node["is_leaf"]:
            continue
        for child in (int(node["left"]), int(node["right"])):
            mid = -(depth[idx] + 0.5)
            ax.plot(
                [x[idx], x[idx], x[child], x[child]],
                [-depth[idx], mid, mid, -depth[child]],
                color=INK, alpha=0.28, lw=0.8, solid_joinstyle="round", zorder=1,
            )

    for idx, node in enumerate(nodes):
        px, py = x[idx], -depth[idx]
        if node["is_leaf"]:
            ax.scatter([px], [py], marker="s", s=26, facecolor="#e4e4e4",
                       edgecolor=MUTED, linewidths=0.8, zorder=3)
        else:
            ax.scatter([px], [py], marker="o", s=26, facecolor=PAPER,
                       edgecolor=INK, linewidths=0.9, zorder=3)
            if depth[idx] <= LABEL_DEPTH:
                feature = names[int(node["feature_idx"])]
                text = (
                    f"{feature} <= {node['num_threshold']:.1f}"
                    if feature == "age"
                    else feature
                )
                ax.annotate(
                    text, (px, py), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=6.5, color=MUTED, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.18", fc=PAPER, ec="none"),
                )

    ax.set_xlim(-1.5, n_leaves + 0.5)
    ax.set_ylim(-max_depth - 1.2, 1.6)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=14)
    ax.text(
        0.0, 1.008,
        f"{sum(1 for n in nodes if not n['is_leaf'])} splits · {n_leaves} leaves "
        f"· max depth {max_depth}",
        transform=ax.transAxes, fontsize=7.5, color=SOFT,
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", mfc=PAPER, mec=INK, ms=5, label="split node"),
            Line2D([], [], marker="s", ls="", mfc="#e4e4e4", mec=MUTED, ms=5, label="leaf"),
        ],
        loc="lower right", frameon=False, fontsize=7.5, labelcolor=MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    algo = build("watch_threshold", load_dataset("data"))
    names = feature_names(algo)
    predictors = algo._model._predictors
    for i, group in enumerate(predictors, start=1):
        render(
            group[0].nodes, names,
            f"watch_threshold · booster tree {i} of {len(predictors)}  "
            f"(P(watch_seconds > {algo.threshold_seconds}))",
            OUT / f"tree-{i:03d}.png",
        )
        if i % 20 == 0:
            print(f"  rendered {i}/{len(predictors)}")

    # Contact sheet: every tree's shape at a glance.
    cols, rows = 10, 10
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20), dpi=90)
    fig.patch.set_facecolor(PAPER)
    for i, ax in enumerate(axes.ravel()):
        ax.set_facecolor(PAPER)
        ax.axis("off")
        if i >= len(predictors):
            continue
        nodes = predictors[i][0].nodes
        depth, x, n_leaves = layout(nodes)
        for idx, node in enumerate(nodes):
            if node["is_leaf"]:
                continue
            for child in (int(node["left"]), int(node["right"])):
                mid = -(depth[idx] + 0.5)
                ax.plot(
                    [x[idx], x[idx], x[child], x[child]],
                    [-depth[idx], mid, mid, -depth[child]],
                    color=INK, alpha=0.5, lw=0.5,
                )
        ax.set_title(f"{i + 1}", fontsize=7, color=SOFT, pad=2)
    fig.suptitle(
        "watch_threshold · all 100 booster trees", fontsize=15, color=INK, y=0.995
    )
    fig.tight_layout()
    fig.savefig(OUT / "all-trees-contact-sheet.png", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {len(predictors)} trees + contact sheet to {OUT}/")


if __name__ == "__main__":
    main()
