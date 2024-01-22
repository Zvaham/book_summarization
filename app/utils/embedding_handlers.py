import numpy as np
from sklearn.cluster import KMeans
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def get_embedding_clusters(docs, num_clusters):
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents([x.page_content for x in docs])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(vectors)

    closest_indices = []
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)

    return selected_indices


if __name__ == "__main__":
    pass
