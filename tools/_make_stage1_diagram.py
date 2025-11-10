# -*- coding: utf-8 -*-
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out_dir = r'outputs/moco_pretrain_stage1/visualizations'
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

palette = {
    'query': '#dce6ff',
    'key': '#fdeeee',
    'queue': '#fdf3d9',
    'loss': '#eaf8e6'
}

def add_box(x, y, width, height, text, color, fontsize=11, weight='normal'):
    box = FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.25', fc=color, ec='#4a4a4a', lw=1.2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, weight=weight)


def add_arrow(start, end, text=None, color='#4a4a4a', linestyle='-', arrowstyle='->'):
    arrow = FancyArrowPatch(start, end, arrowstyle=arrowstyle, mutation_scale=12, lw=1.2, color=color, linestyle=linestyle)
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.04, text, ha='center', va='center', fontsize=10, color=color)

ax.text(0.05, 0.9, 'Stage 1: MoCo v3 Contrastive Pretraining', fontsize=16, weight='bold', ha='left')

add_box(0.05, 0.62, 0.46, 0.24, 'Query Encoder', palette['query'], fontsize=13, weight='bold')
add_box(0.07, 0.68, 0.12, 0.1, 'Block 1', '#ffffff')
add_box(0.21, 0.68, 0.12, 0.1, 'Block 2', '#ffffff')
add_box(0.35, 0.68, 0.12, 0.1, 'Block 3', '#ffffff')
add_box(0.49, 0.68, 0.12, 0.1, 'Projection\nHead', '#ffffff')
add_arrow((0.19, 0.72), (0.21, 0.72))
add_arrow((0.33, 0.72), (0.35, 0.72))
add_arrow((0.47, 0.72), (0.49, 0.72))
ax.text(0.07, 0.8, 'Low-light\nImage', ha='center', va='center', fontsize=10)
add_arrow((0.12, 0.77), (0.12, 0.68))

add_box(0.05, 0.3, 0.46, 0.24, 'Key Encoder (Momentum)', palette['key'], fontsize=13, weight='bold')
add_box(0.07, 0.36, 0.12, 0.1, 'Block 1', '#ffffff')
add_box(0.21, 0.36, 0.12, 0.1, 'Block 2', '#ffffff')
add_box(0.35, 0.36, 0.12, 0.1, 'Block 3', '#ffffff')
add_box(0.49, 0.36, 0.12, 0.1, 'Projection\nHead', '#ffffff')
add_arrow((0.19, 0.4), (0.21, 0.4))
add_arrow((0.33, 0.4), (0.35, 0.4))
add_arrow((0.47, 0.4), (0.49, 0.4))
ax.text(0.07, 0.48, 'Low-light\nImage', ha='center', va='center', fontsize=10)
add_arrow((0.12, 0.45), (0.12, 0.36))

add_arrow((0.55, 0.72), (0.55, 0.44), text='Momentum update', color='#1f77b4')
add_arrow((0.55, 0.44), (0.55, 0.72), color='#1f77b4', arrowstyle='-|>', linestyle=':')

add_box(0.64, 0.32, 0.18, 0.16, 'Dynamic Feature\nQueue', palette['queue'], fontsize=12)
add_arrow((0.61, 0.4), (0.64, 0.4))
add_arrow((0.82, 0.4), (0.9, 0.4))
add_box(0.9, 0.32, 0.08, 0.12, 'Contrastive\nLoss', palette['loss'], fontsize=12)
add_arrow((0.55, 0.72), (0.9, 0.44), color='#7f7f7f')
add_arrow((0.73, 0.4), (0.9, 0.35))

legend_text = 'Legend:\nBlue arrow: Momentum update\nSolid arrow: Forward features\nDashed arrow: Parameter copy'
add_box(0.05, 0.05, 0.32, 0.14, legend_text, '#f4f4f4', fontsize=9)

ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1)
fig.tight_layout()
output_path = os.path.join(out_dir, 'stage1_structure.png')
fig.savefig(output_path, dpi=300)
print(f'Saved diagram to {output_path}')
