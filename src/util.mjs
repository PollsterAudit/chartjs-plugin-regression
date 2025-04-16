
/**
 * Utility class containing the main utility methods
 */
export class Util {
    static transpose(matrix) {
        return matrix[0].map((_, i) => matrix.map(row => row[i]));
    }

    static multiply(A, B) {
        const result = Array(A.length).fill(0).map(() => Array(B[0].length).fill(0));
        for (let i = 0; i < A.length; i++) {
            for (let j = 0; j < B[0].length; j++) {
                for (let k = 0; k < A[0].length; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    // Solve a system of linear equations A x = b using Gauss-Jordan elimination.
    // A is assumed to be a square matrix and b an array.
    static gaussJordanSolve(A, b) {
        const n = A.length;
        // Create augmented matrix [A | b]
        for (let i = 0; i < n; i++) A[i] = [...A[i], b[i]];

        // Forward elimination
        for (let i = 0; i < n; i++) {
            // Find pivot – the row with the largest absolute value in column i.
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
                    maxRow = k;
                }
            }
            // Swap maximum row with current row (pivot within M)
            [A[i], A[maxRow]] = [A[maxRow], A[i]];

            // Check for zero pivot (singular matrix)
            if (Math.abs(A[i][i]) < 1e-10) {
                throw new Error("Matrix is singular or nearly singular");
            }

            // Normalize the pivot row.
            const divisor = A[i][i];
            for (let j = i; j <= n; j++) {
                A[i][j] /= divisor;
            }

            // Eliminate the pivot column entries in other rows.
            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = A[k][i];
                    for (let j = i; j <= n; j++) {
                        A[k][j] -= factor * A[i][j];
                    }
                }
            }
        }

        return A.map(row => row[n]);
    }
}