"""Paper Eq. 8 assignment followed by Eq. 9 online centroid updates."""
import math


class IntentClusterer:
    def __init__(self, num_clusters, update_interval=100, initialization="first_k",
                 update_mode="immediate_eq9"):
        if (num_clusters < 1 or update_interval < 1 or initialization != "first_k"
                or update_mode not in {"immediate_eq9", "buffered"}):
            raise ValueError("Invalid cluster configuration")
        self.num_clusters = num_clusters
        self.update_interval = update_interval
        self.update_mode = update_mode
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

    def assign_with_distance(self, embedding):
        if not self.centroids:
            raise RuntimeError("Initialize on warmup embeddings first")
        if len(embedding) != len(self.centroids[0]) or not all(map(math.isfinite, embedding)):
            raise ValueError("Invalid embedding")
        squared = [sum((a-b)**2 for a, b in zip(embedding, centroid)) for centroid in self.centroids]
        cluster = min(range(self.num_clusters), key=squared.__getitem__)
        return cluster, math.sqrt(squared[cluster])

    def assign(self, embedding):
        return self.assign_with_distance(embedding)[0]

    def update(self, cluster_id, embedding):
        self.assign(embedding)  # dimensional validation
        if not 0 <= cluster_id < self.num_clusters:
            raise ValueError("Unknown cluster")
        n = self.counts[cluster_id]
        self.centroids[cluster_id] = [(n*m+x)/(n+1) for m, x in zip(self.centroids[cluster_id], embedding)]
        self.counts[cluster_id] += 1

    def observe(self, embedding):
        return self.observe_with_diagnostics(embedding)["cluster_id"]

    def observe_with_diagnostics(self, embedding):
        """Assign against current centroids, then update according to the mode."""
        c, distance = self.assign_with_distance(embedding)
        before_count = self.counts[c]
        before_centroid = list(self.centroids[c])
        self.queries += 1
        applied = False
        if self.update_mode == "immediate_eq9":
            self.update(c, embedding)
            applied = True
        else:
            self.pending.append((c, list(embedding)))
            if self.queries % self.update_interval == 0:
                self.flush()
        shift = math.dist(before_centroid, self.centroids[c])
        return dict(cluster_id=c, nearest_centroid_distance_pre_update=distance,
                    centroid_update_applied=applied,
                    cluster_count_before=before_count,
                    cluster_count_after=self.counts[c], centroid_shift_l2=shift)

    def flush(self):
        for c, embedding in self.pending:
            self.update(c, embedding)
        self.pending.clear()
