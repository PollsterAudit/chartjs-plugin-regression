const TerserPlugin = require('terser-webpack-plugin');
const path = require('path');

module.exports = {
    entry: ['./src/index.js', './src/weights.mjs', './src/util.mjs', './src/calculator.mjs'],
    optimization: {
        minimizer: [
            new TerserPlugin({
                terserOptions: {
                    keep_classnames: true
                }
            })
        ]
    },
    output: {
        filename: 'chartjs-plugin-regression-trendline.min.js',
        path: path.resolve(__dirname, 'dist')
    }
};