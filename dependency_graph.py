from argparse import ArgumentParser
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd


p = ArgumentParser()
p.add_argument("heatmap_path")

args = p.parse_args()

heatmap = pd.read_csv(args.heatmap_path, index_col=0)
assert heatmap.index.equals(heatmap.columns)
encodings = heatmap.columns

G = nx.Graph()
edges = []
for enc1, enc2 in itertools.product(list(range(len(encodings))), repeat=2):
  if enc1 == enc2:
    continue
  enc1_name = encodings[enc1]
  enc2_name = encodings[enc2]

  score_forward = heatmap.loc[enc1_name, enc2_name]
  score_reverse = heatmap.loc[enc2_name, enc1_name]

  # Extrema along rows, ignoring diagonal element
  enc1_min = min(heatmap.iloc[enc1, 0:enc1].min() if enc1 > 0 else np.inf, heatmap.iloc[enc1, enc1+1:].min())
  enc2_max = max(heatmap.iloc[enc2, 0:enc2].max() if enc2 > 0 else -np.inf, heatmap.iloc[enc2, enc2+1:].max())
  if score_forward > score_reverse and enc1_min >= enc2_max:
    edges.append((enc1_name, enc2_name))

print(edges)
G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, arrows=True)
plt.savefig("graph.png")
