import {RegressionCalculator} from "./calculator.mjs";

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
            const clearDatasetCache = () => chart.regressionTrendlineCache &&
                (chart.regressionTrendlineCache[datasetIndex] = null);
            const rawData = dataset.data;
            if (!rawData?.length) {
                clearDatasetCache();
                return;
            }

            if (dataset.regressionTrendline && !dataset.regressionTrendline.showLine) {
                clearDatasetCache();
                return;
            }

            // Must be at least 3 points
            if (!Array.isArray(rawData) || rawData.length < 3) {
                clearDatasetCache();
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
                clearDatasetCache();
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
                const y = regression(x);
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
            const hidden = pluginOptions.hidden || !chart.isDatasetVisible(datasetIndex);
            if (hidden) {
                return;
            }

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
}

export { regressionPlugin }
