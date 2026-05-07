import matplotlib.pyplot as plt
import os


def plot_path(
    maze,
    solution,
    start,
    goal,
    moves
):

    x, y = start

    raw_path = [(x, y)]

    for step in solution:

        dx, dy = moves[int(step)]

        nx = x + dx
        ny = y + dy

        if (
            0 <= nx < len(maze)
            and
            0 <= ny < len(maze)
            and
            maze[nx][ny] == 0
        ):

            x, y = nx, ny

            raw_path.append((x, y))

        if (x, y) == goal:
            break

    simplified_path = []

    for position in raw_path:

        if position in simplified_path:

            index = simplified_path.index(position)

            simplified_path = simplified_path[:index + 1]

        else:
            simplified_path.append(position)

    path_x = [p[0] for p in simplified_path]
    path_y = [p[1] for p in simplified_path]

    plt.figure(figsize=(8, 8))

    plt.imshow(
        maze,
        cmap="gray_r"
    )

    plt.plot(
        path_y,
        path_x,
        color="black",
        linewidth=6,
        solid_capstyle="round"
    )

    plt.scatter(
        start[1],
        start[0],
        s=200,
        color="green",
        label="Start"
    )

    plt.scatter(
        goal[1],
        goal[0],
        s=200,
        color="red",
        label="Goal"
    )

    plt.title("Maze Solution Path")

    plt.legend()

    os.makedirs(
        "results/task1",
        exist_ok=True
    )

    plt.savefig(
        "results/task1/maze_solution.png"
    )

    plt.close()