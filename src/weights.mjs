/**
 * Utility class containing methods to help with calculating proper weights. <br>
 * Use this if you plan on rendering LOWESS (Locally Weighted Linear Regression) charts instead of LOESS <p>
 * https://en.wikipedia.org/wiki/Local_regression#Weight_function
 */
export class WeightStrategies {

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

if (typeof window !== 'undefined' && !window.WeightStrategies) {
    window.WeightStrategies = WeightStrategies; // Webpack hack to access the class in the browser
}