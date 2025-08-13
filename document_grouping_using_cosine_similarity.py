import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import faiss

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def _avg_pairwise_similarity(embs: np.ndarray, idxs: List[int]) -> float:
    if len(idxs) <= 1: return 1.0
    V = embs[idxs]  # assume normalized
    sims = V @ V.T
    iu = np.triu_indices(len(idxs), k=1)
    return float(sims[iu].mean()) if iu[0].size else 1.0

def _representative(embs: np.ndarray, idxs: List[int]) -> int:
    if len(idxs) == 1: return idxs[0]
    V = embs[idxs]
    sims = V @ V.T
    scores = sims.mean(axis=1)
    return idxs[int(scores.argmax())]

def group_near_duplicates(
    vectors_dir: str,
    threshold: float = 0.95,
    max_neighbors: int = 20
) -> Dict[str, Any]:
    vdir = Path(vectors_dir)
    embs = np.load(vdir / "embeddings.npy").astype("float32")
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    k = min(max_neighbors, len(embs))
    D, I = index.search(embs, k)

    dsu = DSU(len(embs))
    for i in range(len(embs)):
        for j_idx, s in zip(I[i], D[i]):
            if j_idx != i and s >= threshold:
                dsu.union(i, j_idx)

    comps = {}
    for i in range(len(embs)):
        root = dsu.find(i)
        comps.setdefault(root, []).append(i)

    groups = {}
    for gnum, idxs in enumerate(sorted(comps.values(), key=len, reverse=True), start=1):
        if len(idxs) < 2:
            continue
        rep_idx = _representative(embs, idxs)
        avg_sim = _avg_pairwise_similarity(embs, idxs)
        groups[f"group_{gnum}"] = {
            "members": [meta[i]["page_id"] for i in idxs],
            "similarity": round(avg_sim * 100, 2),
            "rep": meta[rep_idx]["page_id"],
            "size": len(idxs),
        }
    return {"groups": groups, "count": len(embs), "threshold": threshold, "max_neighbors": k}

def summarize_groups(grouped: Dict[str, Any]) -> str:
    return "\n".join(
        f"{gid}: size={g['size']} sim~{g['similarity']}% rep={g['rep']} -> {g['members']}"
        for gid, g in grouped["groups"].items()
    )
