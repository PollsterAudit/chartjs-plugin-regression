
/**
 * Utility class containing the main utility methods
 */
export class Util {
    static transpose(matrix) {
        return matrix[0].map((_, i) => matrix.map(row => row[i]));
    }

    static multiply(A, B) {
        const result = Array(A.length).fill(0).map(() => Array(B[0].length).fill(0));
        for (let i = 0; i < A.length; i++)
            for (let j = 0; j < B[0].length; j++)
                for (let k = 0; k < A[0].length; k++)
                    result[i][j] += A[i][k] * B[k][j];
        return result;
    }

    static gaussJordanSolve(A, b) {
        const n = A.length;
        for (let i = 0; i < n; i++) A[i] = [...A[i], b[i]];

        for (let i = 0; i < n; i++) {
            let maxRow = i;
            for (let k = i + 1; k < n; k++) {
                if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
            }
            [A[i], A[maxRow]] = [A[maxRow], A[i]];

            const divisor = A[i][i];
            for (let j = 0; j <= n; j++) A[i][j] /= divisor;

            for (let k = 0; k < n; k++) {
                if (k !== i) {
                    const factor = A[k][i];
                    for (let j = 0; j <= n; j++) {
                        A[k][j] -= factor * A[i][j];
                    }
                }
            }
        }

        return A.map(row => row[n]);
    }
}