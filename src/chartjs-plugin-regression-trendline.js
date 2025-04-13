/**
 * Regression Trendline Plugin.
 *
 * @author Fx Morin
 */
const regressionPlugin = {
    id: 'regressionTrendline',

    afterDatasetsUpdate(chart) {
        const pluginOptions = chart.options.plugins?.regressionTrendline;
        if (!pluginOptions?.enabled) {
            chart.regressionTrendlineCacheType = null
            chart.regressionTrendlineCache = null;
            return;
        }

        const span = pluginOptions.span || 0.3;

        let zoomBounds = null;
        if ("getZoomedScaleBounds" in chart) { // Zoom plugin support
            zoomBounds = chart.getZoomedScaleBounds();
        }

        chart.data.datasets.forEach((dataset, datasetIndex) => {
            const rawData = dataset.data;
            if (!rawData?.length) {
                if (chart.regressionTrendlineCache) {
                    chart.regressionTrendlineCache[datasetIndex] = null;
                }
                return;
            }

            if (dataset.regressionTrendline && !dataset.regressionTrendline.showLine) {
                if (chart.regressionTrendlineCache) {
                    chart.regressionTrendlineCache[datasetIndex] = null;
                }
                return;
            }

            // Must be at least 3 points
            if (!Array.isArray(rawData) || rawData.length < 3) {
                if (chart.regressionTrendlineCache) {
                    chart.regressionTrendlineCache[datasetIndex] = null;
                }
                return;
            }

            // If a plugin option "weightField" is given, we try to use it; otherwise default weight is 1.
            const dataPoints = rawData.map(pt => ({
                x: typeof pt.x === 'number' ? pt.x : Number(pt[0]),
                y: typeof pt.y === 'number' ? pt.y : Number(pt[1]),
                weight: pluginOptions.weightField && pt[pluginOptions.weightField] != null
                    ? Number(pt[pluginOptions.weightField])
                    : 1
            }));

            const uniqueXPoints = [];
            for (const point of dataPoints) {
                if (!uniqueXPoints.includes(point.x)) {
                    if (zoomBounds !== null && zoomBounds.x !== null &&
                        (point.x < zoomBounds.x.min || point.x > zoomBounds.x.max)) {
                        continue;
                    }
                    uniqueXPoints.push(point.x);
                    if (uniqueXPoints.length > 100) {
                        break;
                    }
                }
            }
            if (uniqueXPoints.length <= 1) {
                if (chart.regressionTrendlineCache) {
                    chart.regressionTrendlineCache[datasetIndex] = null;
                }
                return;
            }

            // Use a linear trendline when there are only 3 points
            let type = uniqueXPoints.length < 4 ? 'linear' : (pluginOptions.type || 'local');
            const degree = pluginOptions.degree ?? (type === 'local' ? 2 : 1);
            const steps = pluginOptions.steps || Math.min(50, Math.max(2, uniqueXPoints.length / 2));

            const regression = RegressionCalculator.computeRegression(dataPoints, type, degree, span);

            // Determine dynamic resolution based on chart width
            const xMin = chart.scales.x.min;
            const xMax = chart.scales.x.max;
            const step = (xMax - xMin) / steps;

            const points = [];
            for (let i = -1; i <= steps + 1; i++) {
                const x = xMin + i * step;
                const y = regression.predict(x);
                if (isFinite(y)) {
                    points.push({ x, y });
                }
            }

            chart.regressionTrendlineCacheType = type;
            if (!chart.regressionTrendlineCache) {
                chart.regressionTrendlineCache = [];
            }
            chart.regressionTrendlineCache[datasetIndex] = points;
        });
    },

    beforeDatasetsDraw(chart) {
        const pluginOptions = chart.options.plugins?.regressionTrendline;
        if (!pluginOptions?.enabled) {
            return;
        }

        if (!chart.regressionTrendlineCacheType || !chart.regressionTrendlineCache) {
            return;
        }

        const { ctx } = chart;

        const type = chart.regressionTrendlineCacheType;
        const borderWidth = pluginOptions.borderWidth || 2;

        chart.data.datasets.forEach((dataset, datasetIndex) => {
            const color = pluginOptions.color || (dataset.borderColor || 'rgba(0,0,0,0.6)');

            const points = chart.regressionTrendlineCache[datasetIndex];
            if (!points) {
                return;
            }

            ctx.lineWidth = borderWidth;
            ctx.strokeStyle = color;

            if (type === 'linear' || type === 'exponential' || type === 'logarithmic' || type === 'power') {
                drawLinearTrendline(ctx, points,
                    pt => chart.scales.x.getPixelForValue(pt.x),
                    pt => chart.scales.y.getPixelForValue(pt.y),
                    chart.chartArea
                );
            } else { // Local & Polynomial
                drawCurvedTrendline(ctx, points,
                    pt => chart.scales.x.getPixelForValue(pt.x),
                    pt => chart.scales.y.getPixelForValue(pt.y),
                    chart.chartArea
                );
            }

            ctx.save();
            ctx.restore();
        });
    }
};

/**
 * Utility class containing methods to help with calculating proper weights. <br>
 * Use this if you plan on rendering LOWESS (Locally Weighted Linear Regression) charts instead of LOESS <p>
 * https://en.wikipedia.org/wiki/Local_regression#Weight_function
 */
class WeightStrategies {

    //region Strategies
    // Just returns a fixed value (no weighting)
    static constant = (point) => 1;

    // Uses a property directly (e.g., point.weightValue)
    static property = (key = 'weight') => (point) => point[key] ?? 1;

    //Precision Weighting - Inverse of variance
    static precision = (sizeKey = 'size', errorKey = 'error') => (point) => {
        const s = point[sizeKey] ?? 0;
        const e = point[errorKey] ?? 0;
        if (e <= 0) {
            return s;
        }
        return s / Math.pow(e, 2); // (Margin Of Error)^2 ~ variance
    }

    // Inverse error weighting: higher error → lower weight
    static inverseError = (errorKey = 'error', defaultMoE = 0) => (point) => {
        const e = point[errorKey] ?? defaultMoE;
        return isFinite(e) ? 1 / (1 + e) : 1;
    };

    // Scales weight linearly based on property (e.g., sample size)
    static scaled = (key = 'size', max = 1000) => (point) => {
        const val = point[key] ?? 0;
        return Math.min(val / max, 1);
    };

    // Combines scaled size and inverse error
    static scaledInverseError = (sizeKey = 'size', errorKey = 'error', maxSize = 1000) => (point) => {
        const size = point[sizeKey] ?? 0;
        const error = point[errorKey] ?? 0;
        return (Math.min(size / maxSize, 1)) * (1 / (1 + error));
    };

    // Uses log-based sample size (stable for wide ranges)
    static logScaled = (key = 'size') => (point) => {
        const val = point[key] ?? 0;
        return Math.log(val + 1); // +1 to avoid log(0)
    };

    // Custom function passed by developer
    static custom = (fn) => (point) => {
        try {
            return fn(point);
        } catch {
            return 1;
        }
    };
    //endregion

    //region Normalization Weights
    /**
     * Given all the values, returns normalized weights using Softmax: exp(x / T) / sum(exp(x / T)) <br>
     * Essentially turning the weights into probability distributions, which all sum to 1. <br>
     * https://en.wikipedia.org/wiki/Softmax_function
     */
    static calculateSoftmaxWeights(values) {
        const max = Math.max(...values);
        const exp = values.map(v => Math.exp(v - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(v => v / sum);
    }

    /**
     * Given all the values, returns normalized weights using Z-Score: (x - μ) / σ <br>
     * https://en.wikipedia.org/wiki/Standard_score
     */
    static calculateZScoreWeights(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const std = Math.sqrt(values.map(v => (v - mean) ** 2).reduce((a, b) => a + b) / values.length);
        return values.map(v => (v - mean) / (std || 1));
    }

    /**
     * Given all the values, returns normalized weights using T-Score: (x - μ) / (s / √n) <br>
     * https://en.wikipedia.org/wiki/Student%27s_t-distribution
     */
    static calculateTScoreWeights(values) {
        const n = values.length;
        const mean = values.reduce((a, b) => a + b, 0) / n;
        const std = Math.sqrt(values.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / n);
        return values.map(v => (v - mean) / ((std || 1) / Math.sqrt(n)));
    }

    /**
     * Given all the values, returns normalized weights using Coefficient of Variation: stddev / mean <br>
     * Lower CV = more consistent = higher weight <br>
     * https://en.wikipedia.org/wiki/Coefficient_of_variation
     */
    static calculateCoefficientOfVariationWeights(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const std = Math.sqrt(values.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / values.length);
        const cv = std / (mean || 1);
        return values.map(() => 1 / (1 + Math.abs(cv)));
    }

    /**
     * Given all the values, returns normalized weights using MinMax: (value - min) / (max - min) <br>
     * https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
     */
    static calculateMinMaxWeights(values) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        if (max === min) {
            return values.map(() => 1); // Avoid division by zero
        }
        return values.map(v => (v - min) / (max - min));
    }

    /**
     * Given all the values, returns normalized weights using Studentized Residuals: res / est stddev of residuals <br>
     * https://en.wikipedia.org/wiki/Studentized_residual
     */
    static calculateStudentizedResidualWeights(values, predictions) {
        if (values.length !== predictions.length) {
            throw new Error("Values and predictions arrays must have the same length.");
        }
        const residuals = values.map((v, i) => v - predictions[i]);
        const residualSD = Math.sqrt(residuals.map(r => r ** 2).reduce((a, b) => a + b, 0) / values.length);
        return residuals.map(r => 1 / (1 + Math.abs(r / (residualSD || 1e-6))));
    }

    /**
     * Given all the values, returns normalized weights using Standardized Moment Weighting: moment / (std^order) <br>
     * Also known as skewness (3), or kurtosis (4). <br>
     * https://en.wikipedia.org/wiki/Standardized_moment
     */
    static calculateStandardizedMomentWeights(values, momentOrder = 3) {
        const n = values.length;
        const mean = values.reduce((a, b) => a + b, 0) / n;
        const std = Math.sqrt(values.map(v => (v - mean) ** 2).reduce((a, b) => a + b, 0) / n);

        const moment = values
            .map(v => (v - mean) ** momentOrder)
            .reduce((a, b) => a + b, 0) / n;

        const standardMoment = moment / Math.pow(std || 1e-6, momentOrder);
        return values.map(() => 1 / (1 + Math.abs(standardMoment)));
    }
    //endregion

    //region Utility functions
    /**
     * Normalize weights to [0, 1]
     */
    static normalizeWeights(weights) {
        const max = Math.max(...weights);
        return weights.map(w => w / (max || 1));
    }

    /**
     * Combines multiple weighting functions into one.
     * Each function gets the data point and returns a weight.
     */
    static combineWeightFns(weightFns = [], combiner = (vals) => vals.reduce((a, b) => a * b, 1)) {
        return (point) => {
            return combiner(weightFns.map(fn => fn(point)));
        };
    }

    /**
     * Allows you to use one of the calculating weights functions as a normal weight function. <br>
     * This will use up a lot more memory than doing it any other way.
     */
    static calculateWeightsFn(points, getValueFromPointFn, calculateWeightsFn) {
        const pointToValueMap = {};
        const values = [];
        points.forEach(p => {
            const value = getValueFromPointFn[p];
            pointToValueMap[p] = value;
            values.push(value);
        });
        const weights = calculateWeightsFn(values);
        const weightMap = {};
        points.reverse().forEach(p => {
            weightMap[p] = weights.pop();
        });
        return (point) => weightMap[point];
    }

    /**
     * Safely map and cache weights into data points, at a certain position
     */
    static applyWeights(data, weightFn) {
        return data.map(p => ({ ...p, weight: weightFn(p) }));
    }
    //endregion
}

/**
 * Utility class containing the methods used to calculate the Regression Trendline
 */
class RegressionCalculator {
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
            ? RegressionCalculator.#computeLocalRegression(normalizedData, degree, span)
            : RegressionCalculator.#fallbackRegression(normalizedData, type, degree);

        // Adjust prediction function to account for normalization
        return {
            predict(x) {
                return regression.predict(x - xMin);
            },
            equation: regression.equation
        };
    }

    static #computeLocalRegression(data, type = 'linear', degree = 2, span = 0.3) {
        // For local regression, data is assumed normalized and may contain "weight".
        return {
            predict(x0) {
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
                const XT = RegressionCalculator.#transpose(X);
                const WX = XT.map((row, j) =>
                    row.map((val, i) => val * W[i])
                );
                const XTWX = RegressionCalculator.#multiply(WX, X);
                const XTWY = WX.map((row) =>
                    row.reduce((sum, v, i) => sum + v * Y[i], 0)
                );
                const coeffs = RegressionCalculator.#gaussJordanSolve(XTWX, XTWY);

                return coeffs.reduce((sum, c, i) => sum + c * (x0 ** i), 0);
            },
            equation: `local regression (degree ${degree}, span ${span})`
        };
    }

    static #fallbackRegression(data, type, degree) {
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
                return {
                    predict: () => yMean,
                    equation: `y = ${yMean.toFixed(2)} (flat line)`
                };
            }

            const slope = weightedSum(x.map((xi, i) => (xi - xMean) * (y[i] - yMean))) / denominator;
            const intercept = yMean - slope * xMean;

            return {
                predict: x => slope * x + intercept,
                equation: `y = ${slope.toFixed(2)}x + ${intercept.toFixed(2)}`
            };
        }

        if (type === 'exponential') {
            // Transform y values and use linear regression on the log scale.
            const logY = y.map(v => Math.log(v));
            const lin = RegressionCalculator.#fallbackRegression(
                data.map((p, i) => ({ x: p.x, y: logY[i], weight: weights[i] })),
                'linear'
            );
            return {
                predict: xVal => Math.exp(lin.predict(xVal)),
                equation: `y = exp(${lin.equation})`
            };
        }

        if (type === 'logarithmic') {
            const logX = x.map(v => Math.log(v));
            const lin = RegressionCalculator.#fallbackRegression(
                data.map((p, i) => ({ x: logX[i], y: y[i], weight: weights[i] })),
                'linear'
            );
            return {
                predict: xVal => lin.predict(Math.log(xVal)),
                equation: `y = ${lin.equation} (log x)`
            };
        }

        if (type === 'power') {
            const logX = x.map(v => Math.log(v));
            const logY = y.map(v => Math.log(v));
            const lin = RegressionCalculator.#fallbackRegression(
                data.map((p, i) => ({ x: logX[i], y: logY[i], weight: weights[i] })),
                'linear'
            );
            return {
                predict: xVal => Math.exp(lin.predict(Math.log(xVal))),
                equation: `y = exp(${lin.equation})`
            };
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
            const coeffs = RegressionCalculator.#gaussJordanSolve(XTX, XTY);

            return {
                predict: xVal => coeffs.reduce((sum, c, i) => sum + c * (xVal ** i), 0),
                equation: `y = ${coeffs.map((c, i) => `${c.toFixed(2)}x^${i}`).join(' + ')}`
            };
        }
        throw new Error(`Unsupported regression type: ${type}`);
    }

    static #transpose(matrix) {
        return matrix[0].map((_, i) => matrix.map(row => row[i]));
    }

    static #multiply(A, B) {
        const result = Array(A.length).fill(0).map(() => Array(B[0].length).fill(0));
        for (let i = 0; i < A.length; i++)
            for (let j = 0; j < B[0].length; j++)
                for (let k = 0; k < A[0].length; k++)
                    result[i][j] += A[i][k] * B[k][j];
        return result;
    }

    static #gaussJordanSolve(A, b) {
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

const drawLinearTrendline = (ctx, points, px, py, bounds) => {
    if (points.length < 2) {
        return;
    }

    const clamp = (val, min, max) => Math.max(min, Math.min(max, val));

    ctx.beginPath();
    for (let i = 0; i < points.length - 1; i++) {
        const point = points[i];
        const x = clamp(px(point), bounds.left, bounds.right);
        const y = clamp(py(point), bounds.top, bounds.bottom);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
};

const drawCurvedTrendline = (ctx, points, px, py, bounds) => {
    if (points.length < 4) {
        return;
    }

    const clamp = (val, min, max) => Math.max(min, Math.min(max, val));
    const line = new Path2D();
    const tension = 0.5;

    line.moveTo(
        clamp(px(points[0]), bounds.left, bounds.right),
        clamp(py(points[0]), bounds.top, bounds.bottom)
    );

    for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i - 1] || points[i];
        const p1 = points[i];
        const p2 = points[i + 1] || p1;
        const p3 = points[i + 2] || p2;

        const x1 = px(p1);
        const y1 = py(p1);
        const x2 = px(p2);
        const y2 = py(p2);

        const cp1x = x1 + (x2 - px(p0)) / 6 * tension;
        const cp2x = x2 - (px(p3) - x1) / 6 * tension;
        const cp1y = y1 + (y2 - py(p0)) / 6 * tension;
        const cp2y = y2 - (py(p3) - y1) / 6 * tension;

        line.bezierCurveTo(
            clamp(cp1x, bounds.left, bounds.right),
            clamp(cp1y, bounds.top, bounds.bottom),
            clamp(cp2x, bounds.left, bounds.right),
            clamp(cp2y, bounds.top, bounds.bottom),
            clamp(x2, bounds.left, bounds.right),
            clamp(y2, bounds.top, bounds.bottom)
        );
    }

    ctx.stroke(line);
};

// Special thanks to https://github.com/Makanz/chartjs-plugin-trendline for this v

// If we're in the browser and have access to the global Chart obj, register plugin automatically
if (typeof window !== 'undefined' && window.Chart) {
    if (window.Chart.hasOwnProperty('register')) {
        window.Chart.register(regressionPlugin);
    } else {
        window.Chart.plugins.register(regressionPlugin);
    }
    window.WeightStrategies = WeightStrategies; // Webpack hack to access the class in the browser
}

// Otherwise, try to export the plugin
try {
    module.exports = regressionPlugin;
    module.exports = WeightStrategies;
} catch (e) {}