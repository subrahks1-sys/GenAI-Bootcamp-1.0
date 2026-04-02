import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


# ============================================================================
# PART 1: VECTORS AND MATRICES FUNDAMENTALS
# ============================================================================


print("=" * 60)
print("PART 1: VECTORS AND MATRICES FUNDAMENTALS")
print("=" * 60)


# --- VECTOR: A list of numbers representing a point in space ---
print("\n📌 VECTOR (1D array):")
vector_2d = np.array([3, 4])
vector_3d = np.array([1, -2, 3.5])
vector_embedding = np.array([0.21, -0.44, 0.89, 0.12, -0.67])


print(f"2D Vector: {vector_2d}")
print(f"3D Vector: {vector_3d}")
print(f"Embedding vector (5D): {vector_embedding}")
print(f"Shape: {vector_embedding.shape}")


# --- MATRIX: Collection of vectors (2D array) ---
print("\n📌 MATRIX (2D array):")
matrix_2x2 = np.array([[1, 2], 
                       [3, 4]])


matrix_3x2 = np.array([[1, 2],
                       [3, 4],
                       [5, 6]])


matrix_2x3 = np.array([[1, 2, 3],
                       [4, 5, 6]])


print(f"2×2 Matrix:\n{matrix_2x2}")
print(f"\n3×2 Matrix:\n{matrix_3x2}")
print(f"\n2×3 Matrix:\n{matrix_2x3}")


# --- VISUALIZATION: 2D Vector Space ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))


# Plot 2D vector
axes[0].arrow(0, 0, vector_2d[0], vector_2d[1], head_width=0.2, head_length=0.2, 
              fc='blue', ec='blue')
axes[0].scatter(vector_2d[0], vector_2d[1], color='red', s=100)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-1, 5)
axes[0].set_ylim(-1, 5)
axes[0].set_xlabel('X-axis')
axes[0].set_ylabel('Y-axis')
axes[0].set_title(f'2D Vector: {vector_2d}')
axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)


# Visualize matrix as multiple vectors
matrix_vectors = np.array([[3, 1], [1, 3], [2, 2]])
for i, vec in enumerate(matrix_vectors):
    axes[1].arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1, 
                  fc=f'C{i}', ec=f'C{i}', label=f'Vector {i+1}')
    axes[1].scatter(vec[0], vec[1], color=f'C{i}', s=80)


axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-1, 5)
axes[1].set_ylim(-1, 5)
axes[1].set_xlabel('X-axis')
axes[1].set_ylabel('Y-axis')
axes[1].set_title('Matrix as Collection of Vectors')
axes[1].legend()
axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)


plt.tight_layout()
plt.show()


print("\n💡 KEY INSIGHT: A matrix is just a collection of vectors!")
print("   - Row vector: shape (1, n)")
print("   - Column vector: shape (n, 1)")
print("   - Embedding = vector with semantic meaning")
