# chartjs-plugin-regression-trendline

A plugin for [Chart.js 4](https://www.chartjs.org/) that adds support for local and global regression trendlines, including advanced local polynomial smoothing.

> 📈 Supports linear, polynomial, exponential, logarithmic, power, and local smoothing trendlines  

Check out an example at [https://pollsteraudit.ca](https://pollsteraudit.ca/)  

---

## 📦 Installation

### On a webpage

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-regression-trendline/dist/chartjs-plugin-regression-trendline.min.js"></script>
```
Make sure chart.js is loaded first!  
That's it, now you just need to add it to a chart.

### With NPM

```bash
npm install chart.js chartjs-plugin-regression-trendline
```

Import and register the plugin with Chart.js:

```js
import { Chart, registerables } from 'chart.js';
import regressionTrendline from 'chartjs-plugin-regression-trendline';

Chart.register(...registerables, regressionTrendline);
```

---

## 🛠️ Usage

In your chart configuration, enable the `regressionTrendline` plugin:

```js
const config = {
  type: 'line',
  data: {
    datasets: [
      {
        label: 'My Data',
        data: [...],
        regressionTrendline: {
          showLine: true
        }
      }
    ]
  },
  options: {
    plugins: {
      regressionTrendline: {
        enabled: true,
        type: 'local', // or 'linear', 'exponential', 'logarithmic', 'power', 'polynomial'
        span: 0.5,     // Only applies for 'local'
        degree: 2,     // Only applies for 'local' or 'polynomial'
        steps: 50,     // Optional - Number of points to draw
        weightField: 'weight', // Weight will be 1 if not set
        color: 'rgba(255, 99, 132, 0.8)', // Defaults to dataset color if not set
        borderWidth: 2
      }
    }
  }
};
```

---

## ⚙️ Options

| Option        | Type                                                                               | Description                                                                                       |
|---------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `enabled`     | `boolean`                                                                          | Enable/disable the trendline                                                                      |
| `type`        | `'local' \| 'linear' \| 'exponential' \| 'logarithmic' \| 'power' \| 'polynomial'` | Type of regression                                                                                |
| `span`        | `number (0–1)`                                                                     | For local regression: percentage of data in the neighborhood                                      |
| `degree`      | `number (0–3+)`                                                                    | Degree of polynomial for local regression. 0 = constant, 1 = linear, etc. Not recommended above 3 |
| `steps`       | `number (≥2)`                                                                      | Number of points to use for drawing the trendline                                                 |
| `weightField` | `string`                                                                           | Name of a field in the data to use as weight. Defaults to 1 if not provided                       |
| `color`       | `color`                                                                            | Trendline color. Defaults to dataset color                                                        |
| `borderWidth` | `number`                                                                           | Width of the trendline                                                                            |

---

## 🎯 Dataset-level Options

You can also control the trendline at the dataset level with:

```js
regressionTrendline: {
  showLine: true // Whether to show the trendline for this dataset
}
```

---

## 📚 Weight Strategies

The plugin includes a `WeightStrategies` class with helpful utilities for defining weights in your data. These can be useful for emphasizing or de-emphasizing certain points when using local regression.

Example usage:

```js
const data = [
    {
        "SampleSize": 1500,
        "MarginOfError": 2.5
    },
    ...
];

const weightedData = WeightStrategies.applyWeights(
    data,
    WeightStrategies.combineWeightFns([
        WeightStrategies.logScaled('SampleSize'),
        WeightStrategies.inverseError('MarginOfError', 2.5)
    ])
);

// [{SampleSize: 1500, MarginOfError: 2.5, weight: 2.0896819518952747},...]
```

---

## Support

This plugin works seamlessly with [chartjs-plugin-zoom](https://www.chartjs.org/chartjs-plugin-zoom/latest/), allowing you to zoom and pan while keeping trendlines correctly aligned with your data.    
The plugin was built to work with line graphs, and hasn't been tested with other graph types.

---

## 📜 License

This project is using the [MIT License](LICENSE)

---

## 🙏 Special Thanks

Thanks to [Makanz](https://github.com/Makanz/chartjs-plugin-trendline/) for creating [chartjs-plugin-trendline](https://github.com/Makanz/chartjs-plugin-trendline/).  
It was a good inspiration and example plugin for me to build my plugin.
It only supports linear trendlines, but it does a ton more than what this plugin can. So you should definitely use that plugin over this one for linear trendlines.