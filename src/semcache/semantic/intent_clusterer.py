"""Eq. 8 assignment and Eq. 9 means, with explicit batched update choice."""
import math


class IntentClusterer:
    def __init__(self, num_clusters, update_interval=100, initialization="first_k"):
        if num_clusters < 1 or update_interval < 1 or initialization != "first_k":
            raise ValueError("Invalid cluster configuration")
        self.num_clusters = num_clusters
        self.update_interval = update_interval
        self.centroids = []
        self.counts = []
        self.pending = []
        self.queries = 0

    def initialize(self, embeddings, counts=None):
        vectors = [list(map(float, e)) for e in embeddings]
        if len(vectors) < self.num_clusters:
            raise ValueError("Need at least C warmup vectors")
        vectors = vectors[:self.num_clusters]
        if not vectors[0] or any(len(e) != len(vectors[0]) or not all(map(math.isfinite, e)) for e in vectors):
            raise ValueError("Invalid embedding dimensions/values")
        self.centroids = vectors
        self.counts = list(counts) if counts is not None else [1] * self.num_clusters
        if len(self.counts) != self.num_clusters or any(c < 0 for c in self.counts):
            raise ValueError("Invalid centroid counts")
        self.pending = []
        self.queries = 0

    def assign(self, embedding):
        if not self.centroids:
            raise RuntimeError("Initialize on warmup embeddings first")
        if len(embedding) != len(self.centroids[0]) or not all(map(math.isfinite, embedding)):
            raise ValueError("Invalid embedding")
        return min(range(self.num_clusters), key=lambda c: sum((a-b)**2 for a, b in zip(embedding, self.centroids[c])))

    def update(self, cluster_id, embedding):
        self.assign(embedding)  # dimensional validation
        if not 0 <= cluster_id < self.num_clusters:
            raise ValueError("Unknown cluster")
        n = self.counts[cluster_id]
        self.centroids[cluster_id] = [(n*m+x)/(n+1) for m, x in zip(self.centroids[cluster_id], embedding)]
        self.counts[cluster_id] += 1

    def observe(self, embedding):
        c = self.assign(embedding)
        self.pending.append((c, list(embedding)))
        self.queries += 1
        if self.queries % self.update_interval == 0:
            self.flush()
        return c

    def flush(self):
        for c, embedding in self.pending:
            self.update(c, embedding)
        self.pending.clear()
