# CU-Maths-Assignment-2_-Gram-Schmidt
import numpy as np

class GramSchmidt:
    def __init__(self, vectors):
        """
        Initializes the GramSchmidt class with input vectors after validation.
        """
        self.vectors = self.validate_vectors(vectors)

    def validate_vectors(self, vectors):
        """
        Validates a set of vectors for consistency and requirements.
        """
        vectors = [np.array(vec) for vec in vectors]

        if not vectors:
            raise ValueError("Input vector list is empty.")

        dims = [len(vec) for vec in vectors]
        if len(set(dims)) > 1:
            raise ValueError("All vectors must have the same number of elements.")

        if dims[0] < 10:
            raise ValueError("All vectors must have at least 10 elements.")

        if dims[0] > 10:
            print("Warning: Vectors have more than 10 elements. Using the first 10 elements.")
            vectors = [vec[:10] for vec in vectors]

        for vec in vectors:
            if not np.issubdtype(vec.dtype, np.number):
                raise ValueError("All vector elements must be numeric.")

        for i, vec in enumerate(vectors):
            if np.linalg.norm(vec) == 0:
                raise ValueError(f"Vector {i+1} is a zero vector.")

        seen = set()
        for i, vec in enumerate(vectors):
            vec_tuple = tuple(vec.tolist())
            if vec_tuple in seen:
                raise ValueError(f"Duplicate vector found at index {i+1}.")
            seen.add(vec_tuple)

        # Check for linear independence
        matrix = np.stack(vectors)
        rank = np.linalg.matrix_rank(matrix)
        if rank < len(vectors):
            raise ValueError("The input vectors are not linearly independent.")

        return vectors
def inner_product(self, v, w):
        """
        Computes the inner product of two vectors v and w.
        """
        return np.dot(v, w)

    def do_gram_schmidt(self):
        """
        Applies the Gram-Schmidt process to the input vectors.
        Returns:
            List of orthonormal vectors.
        """
        orthonormal_basis = []
        
        for v in self.vectors:
            w = v.astype(np.float64).copy()
            for u in orthonormal_basis:
                proj = self.inner_product(v, u) * u
                w -= proj
            norm = np.linalg.norm(w)
            if norm > 1e-10:
                orthonormal_basis.append(w / norm)

        return orthonormal_basis

      def verify_gram_schmidt(self, processed_vectors):
        """
        Verifies if a set of vectors follows the Gram-Schmidt process.
        """
        for i in range(len(processed_vectors)):
            for j in range(i + 1, len(processed_vectors)):
                dot_product = self.inner_product(processed_vectors[i], processed_vectors[j])
                if not np.isclose(dot_product, 0, atol=1e-10):
                    return False, "Vectors are not orthogonal."

        for vec in processed_vectors:
            if not np.isclose(np.linalg.norm(vec), 1, atol=1e-10):
                return False, "Vectors are not normalized."

        for i, v in enumerate(self.vectors):
            reconstructed = np.zeros_like(v, dtype=np.float64)
            for u in processed_vectors:
                reconstructed += self.inner_product(v, u) * u
            if not np.allclose(v, reconstructed, atol=1e-10):
                return False, f"Vector {i+1} does not match its projection."

        return True, "Vectors follow the Gram-Schmidt process."


        # Example usage
try:
    # Input vectors (each list is a vector)
    input_vectors = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    ]

    gs = GramSchmidt(input_vectors)
    orthonormal_basis = gs.do_gram_schmidt()

    print("Orthonormal Basis:")
    for vec in orthonormal_basis:
        print(vec)

    # Verify the result
    valid, message = gs.verify_gram_schmidt(orthonormal_basis)
    print(message)

except ValueError as e:
    print(f"Error: {e}")



# Example usage
try:
    # Input vectors (each list is a vector)
    input_vectors = []

    gs = GramSchmidt(input_vectors)
    orthonormal_basis = gs.do_gram_schmidt()

    print("Orthonormal Basis:")
    for vec in orthonormal_basis:
        print(vec)

    # Verify the result
    valid, message = gs.verify_gram_schmidt(orthonormal_basis)
    print(message)

except ValueError as e:
    print(f"Error: {e}")



# Example usage
try:
    # Input vectors (each list is a vector)
    input_vectors = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    gs = GramSchmidt(input_vectors)
    orthonormal_basis = gs.do_gram_schmidt()

    print("Orthonormal Basis:")
    for vec in orthonormal_basis:
        print(vec)

    # Verify the result
    valid, message = gs.verify_gram_schmidt(orthonormal_basis)
    print(message)

except ValueError as e:
    print(f"Error: {e}")



# Example usage
try:
    # Input vectors (each list is a vector)
    input_vectors = [
        [1, 2, 3, 4],
        [1, 2, 3]
    ]

    gs = GramSchmidt(input_vectors)
    orthonormal_basis = gs.do_gram_schmidt()

    print("Orthonormal Basis:")
    for vec in orthonormal_basis:
        print(vec)

    # Verify the result
    valid, message = gs.verify_gram_schmidt(orthonormal_basis)
    print(message)

except ValueError as e:
    print(f"Error: {e}")



# Example usage
try:
    # Input vectors (each list is a vector)
    input_vectors = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ]

    gs = GramSchmidt(input_vectors)
    orthonormal_basis = gs.do_gram_schmidt()

    print("Orthonormal Basis:")
    for vec in orthonormal_basis:
        print(vec)

    # Verify the result
    valid, message = gs.verify_gram_schmidt(orthonormal_basis)
    print(message)

except ValueError as e:
    print(f"Error: {e}")



# Example usage
try:
    # Input vectors (each list is a vector)
    input_vectors = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    gs = GramSchmidt(input_vectors)
    orthonormal_basis = gs.do_gram_schmidt()

    print("Orthonormal Basis:")
    for vec in orthonormal_basis:
        print(vec)

    # Verify the result
    valid, message = gs.verify_gram_schmidt(orthonormal_basis)
    print(message)

except ValueError as e:
    print(f"Error: {e}")



# Example usage
try:
    # Input vectors (each list is a vector)
    input_vectors = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ]

    gs = GramSchmidt(input_vectors)
    orthonormal_basis = gs.do_gram_schmidt()

    print("Orthonormal Basis:")
    for vec in orthonormal_basis:
        print(vec)

    # Verify the result
    valid, message = gs.verify_gram_schmidt(orthonormal_basis)
    print(message)

except ValueError as e:
    print(f"Error: {e}")



# Example usage
try:
    # Input vectors (each list is a vector)
    input_vectors = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, "10"],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]

    gs = GramSchmidt(input_vectors)
    orthonormal_basis = gs.do_gram_schmidt()

    print("Orthonormal Basis:")
    for vec in orthonormal_basis:
        print(vec)

    # Verify the result
    valid, message = gs.verify_gram_schmidt(orthonormal_basis)
    print(message)

except ValueError as e:
    print(f"Error: {e}")
