namespace NeuralNet;
class Game
{
    string[,] board = new string[8, 8];
    int player_x = 0;
    int player_y = 0;
    string[] initBoard =
    {
        "P.H....X",
        "..H.....",
        "..H..H..",
        "..H..H..",
        "..H..H..",
        "..H..H..",
        ".....H..",
        ".....H.."
    };
    public Game()
    {
        
        for (int i = 0; i < initBoard.GetLength(0); i++)
        {
            for (int j = 0; j < initBoard.GetLength(1); j++)
            {
                board[i, j] = Convert.ToString(initBoard[i][j]);
            }
        }
    }

    public (double, string[,], bool) Action(int dx, int dy)
    {
        bool terminal = false;
        if (0 <= player_x + dx && 0 <= player_y + dy && player_x + dx < 8 && player_y + dy < 8)
        {
            double reward = Math.Sqrt(Math.Pow(-player_x, 2) + Math.Pow(7 - player_y, 2)) - Math.Sqrt(Math.Pow(-(player_x + dx), 2) + Math.Pow(7 - (player_y + dy), 2));
            if (board[player_x + dx, player_y + dy] == "H")
            {
                reward = -10.0;
                terminal = true;
            }
            else
            {
                (board[player_x, player_y], board[player_x + dx, player_y + dy]) = (board[player_x + dx, player_y + dy], board[player_x, player_y]);   
            }
            return (reward, board, terminal);
        }
        else
        {
            return (0, board, false);
        }
    }

    public static double[] GetBitboard(string[,] brd)
    {
        List<int> bitboard = new();
        for (int i = 0; i < brd.GetLength(0); i++)
        {
            for (int j = 0; j < brd.GetLength(1); j++)
            {
                if (brd[i, j] == "P")
                {
                    bitboard.Add(0);
                    bitboard.Add(0);
                }
                else if (brd[i, j] == ".")
                {
                    bitboard.Add(0);
                    bitboard.Add(1);
                }
                else if (brd[i, j] == "H")
                {
                    bitboard.Add(1);
                    bitboard.Add(0);
                }
                else if (brd[i, j] == "X")
                {
                    bitboard.Add(1);
                    bitboard.Add(1);
                }
            }
        }
        double[] bitboard_array = new double[bitboard.Count];
        for (int i = 0; i < bitboard.Count; i++)
        {
            bitboard_array[i] = bitboard[i];
        }
        return bitboard_array;
    }

    public double[] GetState()
    {
        return GetBitboard(board);
    }
}