namespace NeuralNet;
class DeepQ
{
    public Random rnd = new Random();
    public Network action_value;
    public Network target_action_value;
    public double learning_rate;
    public double discount_factor;
    public int batch_size;
    public int replay_memory_size;
    public int target_update;
    public double epsilon_start;
    public double epsilon_end;
    public double epsilon_decay;
    public int steps;
    public DeepQ(int no_inputs, int no_outputs, int no_hidden=64) {
        action_value = new Network(no_inputs, no_hidden, no_outputs);
        target_action_value = action_value.Clone();
        learning_rate = 0.00025;
        discount_factor = 0.99;
        batch_size = 32;
        replay_memory_size = 1000;
        target_update = 10;
        epsilon_start = 1;
        epsilon_end = 0.1;
        epsilon_decay = 0.999;
        steps = 100000;
    }

    public void Train(Game game)
    {
        int target_update_count = target_update;
        double epsilon = epsilon_start;
        List<(double[], double[], double, bool, int)> replay_memory = new();
        List<List<int>> actions = new List<List<int>>
        {
            new List<int> { 0, 1 },
            new List<int> { 1, 0 },
            new List<int> { 0, -1 },
            new List<int> { -1, 0 }
        };
        for (int step = 0; step < steps; step++)
        {
            epsilon = Math.Max(epsilon_end, epsilon_decay * epsilon);
            double[] curr_state = game.GetState();
            while (true)
            {
                int dx, dy, action;
                if (rnd.NextDouble() < epsilon)
                {
                    action = rnd.Next(4);
                    dx = actions[action][0];
                    dy = actions[action][1];
                }
                else
                {
                    double[,] current_state = new double[1, curr_state.Length];
                    for (int i = 0; i < curr_state.Length; i++)
                    {
                        current_state[0, i] = curr_state[i];
                    }
                    double[,] output = action_value.Predict(current_state).matrix;
                    double[] flat_output = new double[output.GetLength(1)];
                    for (int i = 0; i < output.GetLength(1); i++)
                    {
                        flat_output[i] = output[0, i];
                    }
                    action = Array.IndexOf(flat_output, flat_output.Max());
                    dx = actions[action][0];
                    dy = actions[action][1];
                }
                double reward;
                string[,] board;
                bool terminal;
                (reward, board, terminal) = game.Action(dx, dy);
                if (terminal)
                {
                    game = new Game();
                    break;
                }
                double[] next_state = Game.GetBitboard(board);
                if (replay_memory.Count < replay_memory_size)
                {
                    replay_memory.Add((curr_state, next_state, reward, terminal, action));   
                }
                List<(double[], double[], double, bool, int)> mini_batch = new();
                List<int> used_indices = new();
                if (replay_memory.Count > batch_size)
                {
                    while (mini_batch.Count < batch_size)
                    {
                        int next_ind = rnd.Next(replay_memory.Count);
                        if (!used_indices.Contains(next_ind)) {
                            used_indices.Add(next_ind);
                            mini_batch.Add(replay_memory[next_ind]);
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < replay_memory.Count; i++)
                    {
                        mini_batch.Add(replay_memory[i]);
                    }
                }
                double[] ys = new double[batch_size];
                int[] acts = new int[batch_size];
                double[,] inputs = new double[batch_size, mini_batch[0].Item1.Length];
                for (int j = 0; j < batch_size; j++)
                {
                    double[] curr, next;
                    double rwd;
                    bool tmnl;
                    int act;
                    (curr, next, rwd, tmnl, act) = mini_batch[j];
                    if (tmnl)
                    {
                        ys[j] = rwd;
                    }
                    else
                    {
                        double[,] next_wrap = new double[1, next.Length];
                        for (int i = 0; i < next.Length; i++)
                        {
                            next_wrap[0, i] = next[i];
                        }
                        double[,] next_reward = target_action_value.Predict(next_wrap).matrix;
                        double[] next_reward_flat = new double[next_reward.GetLength(1)];
                        for (int i = 0; i < next_reward.GetLength(1); i++)
                        {
                            next_reward_flat[i] = next_reward[0, i];
                        }
                        double max_next_reward = next_reward_flat.Max();
                        ys[j] = rwd + discount_factor * max_next_reward;
                    }
                    for (int k = 0; k < curr.Length; k++)
                    {
                        inputs[j, k] = curr[k];
                    }
                    acts[j] = act;
                }
                double[,] action_value_output = action_value.Predict(inputs).matrix;
                for (int i = 0; i < batch_size; i++)
                {
                    int a = acts[i];
                    double y = ys[i];
                    action_value_output[i, a] = y;
                }
                double loss = action_value.Fit(inputs, action_value_output);
                Console.WriteLine(loss);
                target_update_count -= 1;
                if (target_update_count == 0)
                {
                    target_action_value = action_value.Clone();
                    target_update_count = target_update;
                }
            }
        }
    }
}
