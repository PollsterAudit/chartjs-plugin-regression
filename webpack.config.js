const path = require('path');

module.exports = {
    entry: './src/chartjs-plugin-regression-trendline.js',
    output: {
        filename: 'chartjs-plugin-regression-trendline.min.js',
        path: path.resolve(__dirname, 'dist')
    }
};