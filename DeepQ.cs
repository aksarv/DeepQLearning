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
    public int cycles;
    public DeepQ(int no_inputs, int no_outputs, int no_hidden=64) {
        action_value = new Network(no_inputs, no_hidden, no_outputs);
        target_action_value = action_value.Clone();
        learning_rate = 0.00025;
        discount_factor = 0.99;
        batch_size = 32;
        replay_memory_size = 1000;
        target_update = 1000;
        epsilon_start = 1;
        epsilon_end = 0.1;
        epsilon_decay = 0.9999;
        steps = 1;
        cycles = 200;
    }

    public void Train(bool short_train=true)
    {
        Game game = new();
        int target_update_count = target_update;
        int cycle_count = 0;
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
                // With a probability epsilon, choose a random action, otherwise choose the best according to the action value network
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
                // Add the experience to replay memory
                if (replay_memory.Count < replay_memory_size)
                {
                    replay_memory.Add((curr_state, next_state, reward, terminal, action));   
                }
                // Sample a minibatch of experiences from the replay memory
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
                double[] ys = new double[mini_batch.Count];
                int[] acts = new int[mini_batch.Count];
                double[,] inputs = new double[mini_batch.Count, mini_batch[0].Item1.Length];
                // Calculate the reward, factoring in the immediate reward but also anticipated further rewards as given by the target action value network
                for (int j = 0; j < mini_batch.Count; j++)
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
                // Gradient descent on the action value network compared to the rewards, but only considering the loss with respect to the output node of the action taken
                double[,] action_value_output = action_value.Predict(inputs).matrix;
                for (int i = 0; i < mini_batch.Count; i++)
                {
                    int a = acts[i];
                    double y = ys[i];
                    action_value_output[i, a] = y;
                }
                double loss = action_value.Fit(inputs, action_value_output);
                Console.WriteLine(Convert.ToString(step + 1) + "/" + Convert.ToString(steps) + " | " + Convert.ToString(target_update - target_update_count) + "/" + Convert.ToString(target_update) + " | " + Convert.ToString(cycle_count + 1) + "/" + Convert.ToString(cycles) + " | " + Convert.ToString(loss));
                target_update_count -= 1;
                if (target_update_count == 0)
                {
                    cycle_count += 1;
                    if (cycle_count >= cycles)
                    {
                        break;
                    }
                    target_action_value = action_value.Clone();
                    target_update_count = target_update;
                    if (short_train)
                    {
                        break;
                    }
                }
            }
        }
    }

    public (int, int) BestAction(Game game)
    {
        // Get the best next action
        double[] state = game.GetState();
        double[,] state_wrap = new double[1, state.Length];
        for (int i = 0; i < state.Length; i++)
        {
            state_wrap[0, i] = state[i];
        }
        double[,] q_values_wrap = action_value.Predict(state_wrap).matrix;
        double[] q_values = new double[q_values_wrap.GetLength(1)];
        for (int i = 0; i < q_values_wrap.GetLength(1); i++)
        {
            q_values[i] = q_values_wrap[0, i];
        }
        int action = Array.IndexOf(q_values, q_values.Max());
        List<List<int>> actions = new List<List<int>>
        {
            new List<int> { 0, 1 },
            new List<int> { 1, 0 },
            new List<int> { 0, -1 },
            new List<int> { -1, 0 }
        };
        int dx, dy;
        dx = actions[action][0];
        dy = actions[action][1];
        return (dx, dy);
    }
}
