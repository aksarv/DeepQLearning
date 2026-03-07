// Not complete
namespace NeuralNet;

class Network
{
    public Random rnd = new Random();
    public int no_input;
    public int no_hidden;
    public int no_output;
    public int no_batch;
    public Matrix weights_input_hidden;
    public Matrix weights_hidden_output;
    public Matrix bias_input_hidden;
    public Matrix bias_hidden_output;
    public double learning_rate;

    public double NextGaussian(double mean=0, double stdDev=1)
    {
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        double rndStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * rndStdNormal;
    }

    public Network(int inp, int hid, int ou, double lr=0.1)
    {
        no_input = inp;
        no_hidden = hid;
        no_output = ou;
        double[,] weights_input_hidden_array = new double[inp, hid];
        double[,] weights_hidden_output_array = new double[hid, ou];
        for (int i = 0; i < inp; i++)
        {
            for (int j = 0; j < hid; j++)
            {
                weights_input_hidden_array[i, j] = NextGaussian(0, Math.Sqrt(2.0 / Convert.ToDouble(inp + hid)));
            }
        }
        for (int i = 0; i < hid; i++)
        {
            for (int j = 0; j < ou; j++)
            {
                weights_hidden_output_array[i, j] = NextGaussian(0, Math.Sqrt(2.0 / Convert.ToDouble(hid + ou)));
            }
        }
        weights_input_hidden = new Matrix(weights_input_hidden_array);
        weights_hidden_output = new Matrix(weights_hidden_output_array);
        double[,] bias_input_hidden_array = new double[1, hid];
        double[,] bias_hidden_output_array = new double[1, ou];
        for (int j = 0; j < hid; j++)
        {
            bias_input_hidden_array[0, j] = NextGaussian(0, Math.Sqrt(2.0 / Convert.ToDouble(hid)));
        }
        for (int j = 0; j < ou; j++)
        {
            bias_hidden_output_array[0, j] = NextGaussian(0, Math.Sqrt(2.0 / Convert.ToDouble(ou)));
        }
        bias_input_hidden = new Matrix(bias_input_hidden_array);
        bias_hidden_output = new Matrix(bias_hidden_output_array);
        learning_rate = lr;
    }

    public Matrix ForwardInputHidden(double[,] input)
    {
        Matrix matmul = new Matrix(input) % weights_input_hidden;
        Matrix result = matmul + bias_input_hidden.Broadcast(input.GetLength(0));
        Matrix activated = result.Sigmoid();
        return activated;
    }

    public Matrix ForwardHiddenOutput(double[,] input)
    {
        Matrix matmul = new Matrix(input) % weights_hidden_output;
        Matrix result = matmul + bias_hidden_output.Broadcast(input.GetLength(0));
        Matrix activated = result.Sigmoid();
        return activated;
    }

    public Matrix MSEError(Matrix output, Matrix expected)
    {
        double[,] o = output.matrix;
        double[,] e = expected.matrix;
        int rows = o.GetLength(0);
        int cols = o.GetLength(1);
        double[,] result = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = 0.5 * Math.Pow(e[i, j] - o[i, j], 2);
            }
        }
        return new Matrix(result);
    }

    public double Fit(double[,] input_array, double[,] expected_output_array)
    {
        // Forward Pass
        Matrix input = new Matrix(input_array);
        Matrix expected_output = new Matrix(expected_output_array);
        Matrix output_hidden = ForwardInputHidden(input_array);
        Matrix final_output = ForwardHiddenOutput(output_hidden.matrix);
        Matrix mse_error = MSEError(final_output, expected_output);
        double loss = mse_error.FullSum();
        // Backward Pass from Output to Hidden
        Matrix d_loss_d_activated = final_output - expected_output; 
        Matrix d_activated_d_output = final_output * (1 - final_output);
        Matrix d_loss_d_output = d_loss_d_activated * d_activated_d_output;
        Matrix d_loss_d_weights_hidden_output = output_hidden.T() % d_loss_d_output;
        Matrix d_loss_d_bias_hidden_output = d_loss_d_output.Sum();
        Matrix d_loss_d_hidden_output = d_loss_d_output % weights_hidden_output.T();
        // Backward Pass from Hidden to Input
        Matrix d_activated_d_preactivated = output_hidden * (1 - output_hidden);
        Matrix d_loss_d_hidden = d_activated_d_preactivated * d_loss_d_hidden_output;
        Matrix d_loss_d_weights_input_hidden = input.T() % d_loss_d_hidden;
        Matrix d_loss_d_bias_input_hidden = d_loss_d_hidden.Sum();
        // Apply Weight/Bias Deltas
        weights_input_hidden = weights_input_hidden - learning_rate * d_loss_d_weights_input_hidden;
        weights_hidden_output = weights_hidden_output - learning_rate * d_loss_d_weights_hidden_output;
        bias_input_hidden = bias_input_hidden - learning_rate * d_loss_d_bias_input_hidden;
        bias_hidden_output = bias_hidden_output - learning_rate * d_loss_d_bias_hidden_output;
        return loss;
    }

    public Matrix Predict(double[,] input_array)
    {
        Matrix output_hidden = ForwardInputHidden(input_array);
        Matrix final_output = ForwardHiddenOutput(output_hidden.matrix);
        return final_output;
    }

    public Network Clone()
    {
        Network new_network = new Network(this.no_input, this.no_hidden, this.no_output);
        new_network.weights_input_hidden = this.weights_input_hidden.Clone();
        new_network.weights_hidden_output = this.weights_hidden_output.Clone();
        new_network.bias_input_hidden = this.bias_input_hidden.Clone();
        new_network.bias_hidden_output = this.bias_hidden_output.Clone();
        new_network.learning_rate = this.learning_rate;
        return new_network;
    }
}