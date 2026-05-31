import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from networkx.algorithms import community
import numpy as np

df = pd.read_csv(r"C:\Users\asus\Desktop\grapfprogetes\edges1.csv")
G = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr=["relation", "weight"])

# بررسی اینکه گراف خالی نباشد
print(f"تعداد گره‌ها: {G.number_of_nodes()}")
print(f"تعداد یال‌ها: {G.number_of_edges()}")

#اطلاعات ساختاری گراف
info = {
    "تعداد گره‌ها": G.number_of_nodes(),
    "تعداد یال‌ها": G.number_of_edges(),
    "چگالی گراف": round(nx.density(G), 3),
    "متصل بودن گراف": nx.is_connected(G),
    "تعداد مؤلفه‌های متصل": nx.number_connected_components(G),
    "میانگین درجه": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
    "بیشترین درجه": max(dict(G.degree()).values()),
    "کمترین درجه": min(dict(G.degree()).values()),
    "میانگین ضریب خوشه‌بندی": round(nx.average_clustering(G), 3),
}
info_df = pd.DataFrame(info.items(), columns=["معیار", "مقدار"])
print(info_df.to_string(index=False))

#تعریف انواع مرکزیت ها
degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G)
closeness_cent = nx.closeness_centrality(G)
#مقادیر مرکزیت را در یک جدول می آورد
centrality_df = pd.DataFrame({
    "degree": degree_cent,
    "betweenness": betweenness_cent,
    "closeness": closeness_cent
}).round(3).sort_values("degree", ascending=False)
print(centrality_df.head(10))

# یافتن اجتماع‌ها
communities = list(community.greedy_modularity_communities(G))
print(f"\nتعداد اجتماع‌های یافت شده: {len(communities)}")

for i, com in enumerate(communities):
    print(f"گروه {i+1} (اندازه {len(com)}): {sorted(com)[:10]}...")  # فقط 10 اول را نشان بده

# ========== رنگ‌بندی گره‌ها بر اساس اجتماع ==========
node_to_community = {}
for i, com in enumerate(communities):
    for node in com:
        node_to_community[node] = i

# تولید پالت رنگی
num_communities = len(communities)
if num_communities == 0:
    print("هیچ اجتماعی یافت نشد!")
    exit()

color_palette = plt.cm.tab20(np.linspace(0, 1, num_communities))
node_colors = [color_palette[node_to_community[node]] for node in G.nodes()]

# اندازه گره‌ها
min_degree = min(dict(G.degree()).values())
max_degree = max(dict(G.degree()).values())
node_sizes = [(degree_cent[n] * 2000) + 200 for n in G.nodes()]

# رنگ‌بندی یال‌ها
color_map = {"family": "#e85d30", "romantic": "#d4537e", "ally": "#1d9e75"}
edge_colors = []
edge_widths = []

for u, v, data in G.edges(data=True):
    rel = data.get("relation", "other")
    edge_colors.append(color_map.get(rel, "#888888"))
    edge_widths.append(data.get("weight", 1) * 0.5)

# رسم گراف
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42, k=3)

# رسم گره‌ها
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='black', linewidths=0.5)

# رسم برچسب‌ها
nx.draw_networkx_labels(G, pos, font_size=9, font_color='black', font_weight='bold')

# رسم یال‌ها
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7)


plt.title(f"Kelidar Network - {num_communities} Communities Detected", fontsize=16, pad=20)
plt.axis("off")
plt.tight_layout()

# نمایش گراف
plt.show()

# ذخیره کردن
plt.savefig("kelidar_network_colored.png", dpi=300, bbox_inches="tight")
print("\nگراف ذخیره شد: kelidar_network_colored.png")
plt.savefig("kelidar_network.png", dpi=600, bbox_inches="tight")

path = nx.shortest_path(G, source="Gol Mohamad", target="Soghi")
print(f"کوتاه‌ترین مسیر: {' → '.join(path)}")
print(f"تعداد گام‌ها: {len(path)-1}")

path = nx.shortest_path(G, source="Zagh Abdol", target="Aslan")
print(f"کوتاه‌ترین مسیر: {' → '.join(path)}")
print(f"تعداد گام‌ها: {len(path)-1}")

bridges = list(nx.bridges(G))
print(f"تعداد پل‌ها: {len(bridges)}")
for u, v in bridges:
    print(f"{u} ↔ {v}")

articulation_points = list(nx.articulation_points(G))
print(f"تعداد نودهای بحرانی: {len(articulation_points)}")

import numpy as np


def classify_network(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # معیارهای شبکه
    avg_clustering = nx.average_clustering(G)
    avg_path = nx.average_shortest_path_length(G)
    density = nx.density(G)

    # معیارهای گراف تصادفی معادل
    p = density
    random_clustering = p
    random_path = np.log(n) / np.log(n * p) if n * p > 1 else float('inf')

    print("=" * 40)
    print(f"خوشه‌بندی شبکه:       {avg_clustering:.3f}")
    print(f"خوشه‌بندی تصادفی:     {random_clustering:.3f}")
    print(f"میانگین مسیر شبکه:    {avg_path:.3f}")
    print(f"میانگین مسیر تصادفی:  {random_path:.3f}")
    print("=" * 40)

    # تشخیص Small World
    if avg_clustering > random_clustering and avg_path <= random_path * 1.5:
        print("✅ شبکه Small World هست")
    else:
        print("❌ شبکه Small World نیست")

    # تشخیص Scale-Free
    degrees = [d for n, d in G.degree()]
    degree_counts = np.bincount(degrees)
    degree_counts = degree_counts[degree_counts > 0]
    if degree_counts.std() > degree_counts.mean():
        print("✅ شبکه Scale-Free هست")
    else:
        print("❌ شبکه Scale-Free نیست")


classify_network(G)

from networkx.algorithms.community import girvan_newman

communities_gn = girvan_newman(G)
top_level = next(communities_gn)

for i, com in enumerate(top_level):
    print(f"گروه {i+1}: {sorted(com)}")

edge_betweenness = nx.edge_betweenness_centrality(G)

eb_df = pd.DataFrame(
    [(u, v, round(b, 3)) for (u, v), b in edge_betweenness.items()],
    columns=["از", "به", "میانه"]
).sort_values("میانه", ascending=False)

print(eb_df.head(10))
