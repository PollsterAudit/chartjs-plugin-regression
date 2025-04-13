const path = require('path');

module.exports = {
    entry: './src/chartjs-plugin-regression.js',
    output: {
        filename: 'chartjs-plugin-regression.min.js',
        path: path.resolve(__dirname, 'dist')
    }
};