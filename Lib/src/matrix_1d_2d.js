let col = 6
let row = 10
let data = new Array(row * col).fill().map((_, i) => i)

console.log(data)
// Reshape the array
let reshapedData = [];
for (let i = 0; i < row; i++) {
    reshapedData[i] = [];
    for (let j = 0; j < col; j++) {
        reshapedData[i][j] = data[i * col + j];
    }
}

console.log(reshapedData);

console.table(reshapedData)

console.log(data.slice(col, col))