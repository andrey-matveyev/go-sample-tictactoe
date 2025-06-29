package main

import "math"

// --- Helper functions for matrix operations ---

// AddVectors adds two vectors element-wise.
func AddVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

// SubtractVectors subtracts one vector from another element-wise.
func SubtractVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

// MultiplyVectors multiplies two vectors element-wise.
func MultiplyVectors(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

// MultiplyScalarVector multiplies a vector by a scalar.
func MultiplyScalarVector(scalar float64, vector []float64) []float64 {
	result := make([]float64, len(vector))
	for i := range vector {
		result[i] = scalar * vector[i]
	}
	return result
}

// DotProduct performs the dot product of two vectors.
func DotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// OuterProduct computes the outer product of two vectors.
func OuterProduct(a, b []float64) [][]float64 {
	rows := len(a)
	cols := len(b)
	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
		for j := range result[i] {
			result[i][j] = a[i] * b[j]
		}
	}
	return result
}

// MultiplyMatrixVector multiplies a matrix by a vector.
func MultiplyMatrixVector(matrix [][]float64, vector []float64) []float64 {
	result := make([]float64, len(matrix))
	for i := range matrix {
		result[i] = DotProduct(matrix[i], vector)
	}
	return result
}

// TransposeMatrix transposes a matrix.
func TransposeMatrix(matrix [][]float64) [][]float64 {
	rows := len(matrix)
	if rows == 0 {
		return [][]float64{}
	}
	cols := len(matrix[0])
	transposed := make([][]float64, cols)
	for i := range transposed {
		transposed[i] = make([]float64, rows)
	}
	for i := range rows {
		for j := range cols {
			transposed[j][i] = matrix[i][j]
		}
	}
	return transposed
}

// --- Activation functions

// Tanh activation function (hyperbolic tangent).
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// TanhDerivative derivative of the Tanh function.
func TanhDerivative(x float64) float64 {
	t := math.Tanh(x)
	return 1 - t*t
}
