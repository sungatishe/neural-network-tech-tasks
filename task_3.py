import numpy as np
import matplotlib.pyplot as plt

# Task 1: Sum of p matrix-vector products
def matrix_vector_sum(file_matrices, file_vectors):
    # Read matrices and vectors from files
    matrices = [np.loadtxt(f'matrix_{i}.txt') for i in range(p)]
    vectors = [np.loadtxt(f'vector_{i}.txt') for i in range(p)]
    p = len(matrices)  # Number of matrix-vector pairs
    n = matrices[0].shape[0]  # Size of square matrices
    result = np.zeros((n, 1))
    for i in range(p):
        result += np.dot(matrices[i], vectors[i])
    np.savetxt('task1_result.txt', result)
    return result

# Task 2: Convert vector to binary matrix
def vector_to_binary(file_vector):
    vector = np.loadtxt(file_vector)
    binary_matrix = np.array([[int(bit) for bit in format(int(x), '08b')] for x in vector])
    np.savetxt('task2_result.txt', binary_matrix, fmt='%d')
    return binary_matrix

# Task 3: Find unique rows in matrix
def unique_rows(file_matrix):
    matrix = np.loadtxt(file_matrix)
    unique_rows = np.unique(matrix, axis=0)
    np.savetxt('task3_result.txt', unique_rows, fmt='%.2f')
    return unique_rows

# Task 4: Matrix with normal distribution, column stats, and row histograms
def normal_matrix_stats(M, N):
    matrix = np.random.normal(0, 1, size=(M, N))
    col_means = np.mean(matrix, axis=0)
    col_variances = np.var(matrix, axis=0)
    np.savetxt('task4_matrix.txt', matrix)
    np.savetxt('task4_means.txt', col_means)
    np.savetxt('task4_variances.txt', col_variances)
    for i in range(M):
        plt.hist(matrix[i], bins=10, alpha=0.5, label=f'Row {i}')
    plt.legend()
    plt.savefig('task4_histograms.png')
    plt.close()
    return matrix, col_means, col_variances

# Task 5: Chessboard pattern matrix
def chessboard_matrix(M, N, a, b):
    matrix = np.zeros((M, N))
    matrix[::2, ::2] = a  # Even rows, even columns
    matrix[1::2, 1::2] = a  # Odd rows, odd columns
    matrix[::2, 1::2] = b  # Even rows, odd columns
    matrix[1::2, ::2] = b  # Odd rows, even columns
    np.savetxt('task5_result.txt', matrix, fmt='%.2f')
    return matrix

# Task 6: Generate circle image tensor
def create_circle_image(height, width, radius, r, g, b):
    center = (height // 2, width // 2)
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = dist_from_center <= radius
    image = np.zeros((height, width, 3))
    image[mask] = [r, g, b]
    np.save('task6_result.npy', image)
    return image

# Task 7: Standardize tensor
def standardize_tensor(file_tensor):
    tensor = np.load(file_tensor + '.npy')
    mean = np.mean(tensor)
    std = np.std(tensor)
    standardized = (tensor - mean) / std
    np.save('task7_result.npy', standardized)
    return standardized

# Task 8: Extract fixed-size patch from matrix
def extract_patch(file_matrix, row, col, size, fill_value):
    matrix = np.loadtxt(file_matrix)
    height, width = matrix.shape
    start_row = max(0, row - size // 2)
    end_row = min(height, row + size // 2 + 1)
    start_col = max(0, col - size // 2)
    end_col = min(width, col + size // 2 + 1)
    patch = np.full((size, size), fill_value)
    patch_row_start = max(0, size // 2 - row)
    patch_col_start = max(0, size // 2 - col)
    patch[patch_row_start:patch_row_start + (end_row - start_row),
          patch_col_start:patch_col_start + (end_col - start_col)] = \
        matrix[start_row:end_row, start_col:end_col]
    np.savetxt('task8_result.txt', patch, fmt='%.2f')
    return patch

# Task 9: Most frequent number per row
def most_frequent_per_row(file_matrix):
    matrix = np.loadtxt(file_matrix)
    most_frequent = np.array([np.bincount(row.astype(int)).argmax() for row in matrix])
    np.savetxt('task9_result.txt', most_frequent, fmt='%d')
    return most_frequent

# Task 10: Combine image channels with weights
def combine_channels(file_image, weights):
    image = np.load(file_image + '.npy')
    height, width, num_channels = image.shape
    combined = np.zeros((height, width))
    for i in range(num_channels):
        combined += image[:, :, i] * weights[i]
    np.save('task10_result.npy', combined)
    return combined

# Example usage (adjust file names and parameters as needed)
if __name__ == "__main__":
    # Task 1 example (assuming p=2, n=3 matrices and vectors)
    p = 2
    matrix_vector_sum('matrices', 'vectors')

    # Task 2 example
    vector_to_binary('vector.txt')

    # Task 3 example
    unique_rows('matrix.txt')

    # Task 4 example
    normal_matrix_stats(5, 5)

    # Task 5 example
    chessboard_matrix(4, 4, 1, 0)

    # Task 6 example
    create_circle_image(100, 100, 20, 1, 0, 0)

    # Task 7 example
    standardize_tensor('tensor')

    # Task 8 example
    extract_patch('matrix.txt', 2, 2, 3, -1)

    # Task 9 example
    most_frequent_per_row('matrix.txt')

    # Task 10 example
    combine_channels('image', [0.3, 0.6, 0.1])