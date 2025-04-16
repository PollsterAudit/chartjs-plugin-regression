const path = require('path');

module.exports = {
    entry: ['./src/index.js', './src/weights.mjs', './src/util.mjs', './src/calculator.mjs'],
    output: {
        filename: 'chartjs-plugin-regression-trendline.min.js',
        path: path.resolve(__dirname, 'dist')
    }
};