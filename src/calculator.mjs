import {Util} from "./util.mjs";

/**
 * Utility class containing the methods used to calculate the Regression Trendline
 */
export class RegressionCalculator {
    static computeRegression(data, type = 'linear', degree = 2, span = 0.3) {
        // Determine the minimum x-value for normalization
        const xMin = Math.min(...data.map(p => p.x));

        // Normalize x-values
        const normalizedData = data.map(p => ({
            ...p,
            x: p.x - xMin
        }));

        // Compute regression on normalized data
        const regression = (type === 'local')
            ? RegressionCalculator.computeLocalRegression(normalizedData, degree, span)
            : RegressionCalculator.fallbackRegression(normalizedData, type, degree);

        // Adjust prediction function to account for normalization
        return (x) => regression(x - xMin);
        // equation: regression.equation
    }

    static computeLocalRegression(data, degree = 2, span = 0.3) {
        // For local regression, data is assumed normalized and may contain "weight".
        return (x0) => {
            // Compute distances using normalized x values.
            const distances = data.map(p => Math.abs(p.x - x0));
            const sorted = distances
                .map((d, i) => ({ index: i, dist: d }))
                .sort((a, b) => a.dist - b.dist);
            const bandwidth = Math.floor(span * data.length);
            const maxDist = sorted[bandwidth - 1]?.dist || 1e-10;

            const tricube = d => {
                const w = 1 - Math.pow(d / maxDist, 3);
                return Math.pow(Math.max(0, w), 3);
            };

            const W = new Array(bandwidth);
            const X = new Array(bandwidth);
            const Y = new Array(bandwidth);

            for (let i = 0; i < bandwidth; i++) {
                const { index } = sorted[i];
                const xi = data[index].x;
                const yi = data[index].y;
                const wCustom = data[index].weight || 1;
                const dist = Math.abs(xi - x0);
                // Multiply the tricube weight with the custom point weight
                W[i] = tricube(dist) * wCustom;
                X[i] = Array.from({ length: degree + 1 }, (_, d) => Math.pow(xi, d));
                Y[i] = yi;
            }

            // Weighted least squares: solve (Xᵗ W X) β = Xᵗ W y
            const XT = Util.transpose(X);
            const WX = XT.map((row, j) =>
                row.map((val, i) => val * W[i])
            );
            const XTWX = Util.multiply(WX, X);
            const XTWY = WX.map((row) =>
                row.reduce((sum, v, i) => sum + v * Y[i], 0)
            );
            const coeffs = Util.gaussJordanSolve(XTWX, XTWY);

            return coeffs.reduce((sum, c, i) => sum + c * (x0 ** i), 0);
        };
        // equation: `local regression (degree ${degree}, span ${span})`
    }

    static fallbackRegression(data, type, degree) {
        // Extract normalized x and y arrays from data.
        const x = data.map(p => p.x);
        const y = data.map(p => p.y);
        const weights = data.map(p => p.weight || 1);

        const weightedSum = (arr) => arr.reduce((sum, v, i) => sum + v * weights[i], 0);
        const sumWeights = weights.reduce((a, b) => a + b, 0);
        const weightedMean = (arr) => weightedSum(arr) / sumWeights;

        if (type === 'linear') {
            const xMean = weightedMean(x);
            const yMean = weightedMean(y);
            const denominator = weightedSum(x.map(xi => (xi - xMean) ** 2));

            if (Math.abs(denominator) < 1e-10) {
                return () => yMean;
                // equation: `y = ${yMean.toFixed(2)} (flat line)`
            }

            const slope = weightedSum(x.map((xi, i) => (xi - xMean) * (y[i] - yMean))) / denominator;
            const intercept = yMean - slope * xMean;

            return x => slope * x + intercept;
            // equation: `y = ${slope.toFixed(2)}x + ${intercept.toFixed(2)}`
        }

        if (type === 'exponential') {
            // Transform y values and use linear regression on the log scale.
            const logY = y.map(v => Math.log(v));
            const lin = RegressionCalculator.fallbackRegression(
                data.map((p, i) => ({ x: p.x, y: logY[i], weight: weights[i] })),
                'linear'
            );
            return xVal => Math.exp(lin(xVal));
            // equation: `y = exp(${lin.equation})`
        }

        if (type === 'logarithmic') {
            const logX = x.map(v => Math.log(v));
            const lin = RegressionCalculator.fallbackRegression(
                data.map((p, i) => ({ x: logX[i], y: y[i], weight: weights[i] })),
                'linear'
            );
            return xVal => lin(Math.log(xVal));
            // equation: `y = ${lin.equation} (log x)`
        }

        if (type === 'power') {
            const logX = x.map(v => Math.log(v));
            const logY = y.map(v => Math.log(v));
            const lin = RegressionCalculator.fallbackRegression(
                data.map((p, i) => ({ x: logX[i], y: logY[i], weight: weights[i] })),
                'linear'
            );
            return xVal => Math.exp(lin(Math.log(xVal)));
            // equation: `y = exp(${lin.equation})`
        }

        if (type === 'polynomial') {
            const X = [], Y = [];
            // For weighted least squares, multiply each row by the square root of its weight
            for (let i = 0; i < x.length; i++) {
                const wSqrt = Math.sqrt(weights[i]);
                X[i] = [];
                for (let j = 0; j <= degree; j++) {
                    X[i][j] = wSqrt * Math.pow(x[i], j);
                }
                Y[i] = wSqrt * y[i];
            }

            const XT = X[0].map((_, j) => X.map(row => row[j]));
            const XTX = XT.map(row => XT.map(col => row.reduce((a, b, i) => a + b * col[i], 0)));
            const XTY = XT.map(row => row.reduce((a, b, i) => a + b * Y[i], 0));
            const coeffs = Util.gaussJordanSolve(XTX, XTY);

            return xVal => coeffs.reduce((sum, c, i) => sum + c * (xVal ** i), 0);
            // equation: `y = ${coeffs.map((c, i) => `${c.toFixed(2)}x^${i}`).join(' + ')}`
        }
        throw new Error(`Unsupported regression type: ${type}`);
    }
}