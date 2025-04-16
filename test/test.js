import { Util } from "../src/util.mjs";
import { RegressionCalculator } from "../src/calculator.mjs";
import assert from "assert";

function testTranspose() {
    const matrix = [
        [1, 2, 3],
        [4, 5, 6]
    ];
    const expected = [
        [1, 4],
        [2, 5],
        [3, 6]
    ];
    const result = Util.transpose(matrix);
    assert.deepStrictEqual(result, expected);
}

function testMultiply() {
    const A = [
        [1, 2],
        [3, 4]
    ];
    const B = [
        [5, 6],
        [7, 8]
    ];
    const expected = [
        [1 * 5 + 2 * 7, 1 * 6 + 2 * 8],
        [3 * 5 + 4 * 7, 3 * 6 + 4 * 8]
    ]; // i.e., [[19, 22], [43, 50]]
    const result = Util.multiply(A, B);
    assert.deepStrictEqual(result, expected);
}

function testLinearSystem() {
    assert.deepEqual([1,2], Util.gaussJordanSolve([[1,2],[3,4]], [5,11]));
}

function testThrowOnInvalidSolution() {
    Util.gaussJordanSolve([[1,0,0],[0,0,1],[0,0,2]], [1,1,1]);
}

function testSimpleGaussJordanSolve() {
    const A = [
        [2, 1],
        [1, 3]
    ];
    const b = [5, 7]; // 2x + y = 5, x + 3y = 7
    const expected = [1.6, 1.8]; // ~ x = 1.6, y = 1.8
    const solution = Util.gaussJordanSolve(A, b);
    // Check each value with a tolerance to handle floating point arithmetic.
    expected.forEach((val, idx) => {
        assert(Math.abs(solution[idx] - val) < 1e-10, `Index ${idx}: expected ${val}, but got ${solution[idx]}`);
    });
}

function testLocalRegression() {
    // For a simple line y = 2 + 3x, let’s build normalized data points.
    const data = [];
    for (let x = 0; x <= 1; x += 0.1) {
        data.push({ x: x, y: 2 + 3 * x });
    }
    // Use linear (degree 1) regression with a span that uses half of the available points.
    const predictor = RegressionCalculator.computeLocalRegression(data, 1, 0.5);
    const testX = 0.5;
    const predictedY = predictor(testX);
    const expectedY = 2 + 3 * testX; // which should be 3.5
    // Allow for a small error tolerance due to numerical approximations
    assert(Math.abs(predictedY - expectedY) < 0.5, `Predicted value ${predictedY} is not close to ${expectedY}`);
}

function testQuadraticFit() {
    // Create quadratic data: y = 1 + 2x + x^2 with a slight random noise.
    const data = [];
    for (let x = 0; x <= 1; x += 0.1) {
        // adding a minor noise (here noise is zeroed out for predictability)
        data.push({ x, y: 1 + 2 * x + x * x });
    }
    // Use quadratic regression (degree 2) with a span that uses about 70% of the points.
    const predictor = RegressionCalculator.computeLocalRegression(data, 2, 0.7);
    const testX = 0.75;
    const predictedY = predictor(testX);
    const expectedY = 1 + 2 * testX + testX * testX;
    // Allow a small tolerance for numeric error.
    assert(Math.abs(predictedY - expectedY) < 0.5, `Predicted value ${predictedY} is not close to ${expectedY}`);
}

testTranspose();
testMultiply();
testLinearSystem();
assert.throws(function(){
    testThrowOnInvalidSolution();
});
testSimpleGaussJordanSolve();
testLocalRegression();
testQuadraticFit();