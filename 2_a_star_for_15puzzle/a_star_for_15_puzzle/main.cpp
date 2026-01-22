#include "puzzle.hpp"
#include "a_star.hpp"
#include <iostream>

int main()
{
	auto goal = init_Goal();
	auto puzzle = init_Puzzle();
	print_Puzzle(puzzle);
	if (!is_Solvable(puzzle))
	{
		std::cout << "Puzzle is NOT solvable" << std::endl;
		return 0;
	}
    // Solve the puzzle using A* with Manhattan distance heuristic
    std::cout << "Solving with Manhattan Distance Heuristic..." << std::endl;
    auto path = a_star(puzzle, goal, h_mnhtn);

    // Check if a solution was found
    if (path.empty())
    {
        std::cout << "No solution found!" << std::endl;
        return 0;
    }

    // Print the solution path
    std::cout << "Solution Path:" << std::endl;
    for (const auto& step : path)
    {
        print_Puzzle(step);
        std::cout << "---" << std::endl;
    }

    return 0;

}