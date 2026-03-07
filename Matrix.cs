using System.Reflection.Metadata;

namespace NeuralNet;

class Matrix
{
    public double[,] matrix;
    public Matrix(double[,] input)
    {
        matrix = input;
    }

    public static Matrix operator +(Matrix left, Matrix right)
    {
        if (left.matrix.GetLength(0) != right.matrix.GetLength(0) || left.matrix.GetLength(1) != right.matrix.GetLength(1))
        {
            throw new Exception("Matrix addition failed as dimensions are not equal");
        }
        int rows = left.matrix.GetLength(0);
        int cols = left.matrix.GetLength(1);
        if (right.matrix.GetLength(0) == 1)
        {
            double[,] result1 = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result1[i, j] = left.matrix[i, j] + right.matrix[0, j];
                }
            }
            return new Matrix(result1);
        }
        double[,] result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = left.matrix[i, j] + right.matrix[i, j];
            }
        }
        return new Matrix(result);
    }

    public static Matrix operator -(Matrix left, Matrix right)
    {
        if (left.matrix.GetLength(0) != right.matrix.GetLength(0) || left.matrix.GetLength(1) != right.matrix.GetLength(1))
        {
            throw new Exception("Matrix subtraction failed as dimensions are not equal");
        }
        int rows = left.matrix.GetLength(0);
        int cols = right.matrix.GetLength(1);
        double[,] result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = left.matrix[i, j] - right.matrix[i, j];
            }
        }
        return new Matrix(result);
    }
    
    public static Matrix operator -(int left, Matrix right)
    {
        int rows = right.matrix.GetLength(0);
        int cols = right.matrix.GetLength(1);
        double[,] result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = left - right.matrix[i, j];
            }
        }
        return new Matrix(result);
    }

    public static Matrix operator *(Matrix left, Matrix right)
    {
        if (left.matrix.GetLength(0) != right.matrix.GetLength(0) || left.matrix.GetLength(1) != right.matrix.GetLength(1))
        {
            throw new Exception("Matrix elementwise multiplication failed as dimensions are not equal");
        }
        int rows = left.matrix.GetLength(0);
        int cols = right.matrix.GetLength(1);
        double[,] result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = left.matrix[i, j] * right.matrix[i, j];
            }
        }
        return new Matrix(result);
    }

    public static Matrix operator *(double left, Matrix right)
    {
        int rows = right.matrix.GetLength(0);
        int cols = right.matrix.GetLength(1);
        double[,] result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = left * right.matrix[i, j];
            }
        }
        return new Matrix(result);
    }

    public static double Dot(double[] v1, double[] v2)
    {
        if (v1.Length != v2.Length)
        {
            throw new Exception("Dot product failed as vectors are not the same length");
        }
        double result = 0;
        for (int i = 0; i < v1.Length; i++)
        {
            result += v1[i] * v2[i];
        }
        return result;
    }

    public static Matrix operator %(Matrix m1, Matrix m2)
    {
        if (m1.matrix.GetLength(1) != m2.matrix.GetLength(0))
        {
            throw new Exception("Matrix multiplication failed as matrices are not conformable");
        }
        double[,] result = new double[m1.matrix.GetLength(0), m2.matrix.GetLength(1)];
        for (int i = 0; i < m1.matrix.GetLength(0); i++)
        {
            for (int j = 0; j < m2.matrix.GetLength(1); j++)
            {
                double[] row = new double[m1.matrix.GetLength(1)];
                for (int k = 0; k < m1.matrix.GetLength(1); k++)
                {
                    row[k] = m1.matrix[i, k];
                }
                double[] col = new double[m2.matrix.GetLength(0)];
                for (int k = 0; k < m2.matrix.GetLength(0); k++)
                {
                    col[k] = m2.matrix[k, j];
                }
                result[i, j] = Dot(row, col);
            }
        }
        return new Matrix(result);
    }

    public Matrix Sigmoid()
    {  
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        double[,] result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = 1.0 / (1.0 + Math.Exp(-matrix[i, j]));
            }
        }
        return new Matrix(result);
    }

    public void WriteMatrix()
    {
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            string line = "";
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                line += Convert.ToString(matrix[i, j]) + "\t";
            }
            System.Console.WriteLine(line);
        }
    }

    public Matrix T()
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        double[,] result = new double[cols, rows];
        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                result[i, j] = matrix[j, i];
            }
        }
        return new Matrix(result);
    }

    public Matrix Sum(int axis=1)
    {
        if (axis == 1)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[,] result = new double[1, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[0, j] += matrix[i, j];
                }
            }
            return new Matrix(result);
        }
        else
        {
            throw new Exception("This hasn't been coded yet lmao");
        }
    }

    public double FullSum()
    {
        double full_sum = 0.0;
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                full_sum += matrix[i, j];
            }
        }
        return full_sum;
    }

    public Matrix Broadcast(int n)
    {
        double[,] result = new double[n, matrix.GetLength(1)];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                result[i, j] = matrix[0, j];
            }
        }
        return new Matrix(result);
    }

    public Matrix Clone()
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        double[,] matrix_copy = new double[rows, cols];
        Array.Copy(matrix, matrix_copy, matrix.Length);
        return new Matrix(matrix_copy);
    }
}