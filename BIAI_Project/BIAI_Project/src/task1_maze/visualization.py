import matplotlib.pyplot as plt
import os

def plot_path(maze, solution, start, goal, moves):
    x, y = start
    path_x, path_y = [x], [y]

    for step in solution:
        dx, dy = moves[int(step)]
        nx, ny = x + dx, y + dy

        if 0 <= nx < len(maze) and 0 <= ny < len(maze) and maze[nx][ny] == 0:
            x, y = nx, ny
            path_x.append(x)
            path_y.append(y)

        if (x, y) == goal:
            break

    plt.imshow(maze, cmap='gray')
    plt.plot(path_y, path_x, marker='o')

    os.makedirs("results/task1", exist_ok=True)
    plt.savefig("results/task1/maze_solution.png")

    plt.show()