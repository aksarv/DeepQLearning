namespace NeuralNet;
class Program
{
    static void Main(string[] args)
    {
        DeepQ agent = new(128, 4);
        agent.Train(short_train: false);
        Game game = new();
        int dx, dy;
        (dx, dy) = agent.BestAction(game);
        while (true)
        {
            double reward;
            string[,] board;
            bool terminal; 
            (reward, board, terminal) = game.Action(dx, dy);
            if (terminal)
            {
                Console.WriteLine("Game Over!");
                break;
            }
            for (int i = 0; i < 8; i++)
            {
                string row = "";
                for (int j = 0; j < 8; j++)
                {
                    row += board[i, j];
                }
                Console.WriteLine(row);
            }
            Console.WriteLine();
            Thread.Sleep(1000);
        }
    }
}
