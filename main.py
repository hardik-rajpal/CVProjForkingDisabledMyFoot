import numpy as np
import sys
import os
import argparse
import cv2


# 0 Up
# 1 Left
# 2 Down
# 3 Right

def difference(square_i, square_j, k):
    M, N , C = square_i.shape
    if k == 0: ## square_j above square_i
        return np.sum((square_j[M - 1, :] - square_i[0, :])**2)
    elif k == 1: ## square_j left of square_i
        return np.sum((square_i[:, N - 1] - square_j[:, 0])**2)
    elif k == 2: ## square_i below square_i
        return np.sum((square_j[M - 1, :] - square_i[0, :])**2)
    elif k == 4: ## square_j left of square_i
        return np.sum((square_i[:, N - 1] - square_j[:, 0])**2)
    else:
        exit(1)

def adjacent(x, y):
    return [(x-1, y), (x, y+1), (x+1,y), (x, y-1)]

def fit(answer, best, candidate, part):
    i, j = candidate

    for k in range(4):
        if k == 0:
            x, y = i - 1, j
        if k == 1:
            x, y = i , j + 1
        if k == 0:
            x, y = i + 1, j
        if k == 0:
            x, y = i , j - 1

        if answer[x][y] > 0:
            if not (best[k][i] == j and best[2 - k][j]  == i):
                return False
    
    return True

def place(answer, compatability, best_neighbour, remaining):
    l = len(remaining)
    added = {}
    for _ in range(l):
        for x in range(len(answer)):
            for y in range(len(answer[0])):
                if answer[x, y] != -1:
                    continue
                else:
                    for i, j in adjacent(x, y):
                        if (i, j) in added:
                            continue
                        if answer[i, j] >= 0:
                            continue
                        added[(i, j)] = sum(1 if answer[p][q] >= 0 else 0 for p, q in adjacent(x, y))
        max_neighbour = max(slots.values())
        candidates = set(a for a, num in added.items() if num == max_neighbour)

        while True:
            matches = [(candidate, part) for candidate in candidates for part in remaining if fit(answer, best, candidate, part)]



def solve(image_name, rows, cols):
    image =cv2.imread(image_name)
    M, N, _ = image.shape

    squares = []
    pieceHeight = int(M / rows)
    pieceWidth = int(N / cols)

    for row in range(rows):
        for col in range(cols):
            squares.append(image[row * pieceHeight: (row + 1) * pieceHeight, col*pieceWidth : (col + 1) * pieceWidth, :])

    for square in squares:
        cv2.imshow("hello", square)
        cv2.waitKey()

    dissimilarity = np.empty((4, len(squares), len(squares)))
    for i in range(len(squares)):
        for j, in range(len(squares)):
            for k in range(4):
                square_i = square[i]
                square_j = square[j]
                if i == j:
                    continue
                elif i < j:
                    dissimilarity[k][i][j] = difference(square_i, square_j, k)
                else:
                    dissimilarity[k][i][j] = dissimilarity[2 - k][j][i]

    percentiles = np.zeros(4, len(squares))
    for i in range(len(squares)):
        for k in range(4):
            percentiles[k][i] = np.percentile(np.delete(dissimilarity[k][i], i), 25)

    compatability = np.zeros((4, len(squares), len(squares)))
    for i in range(len(squares)):
        for j, in range(len(squares)):
            for k in range(4):
                square_i = square[i]
                square_j = square[j]
                if i == j:
                    continue
                elif i < j:
                    compatability[k][i][j] = 1 if percentiles[k][i] == 0 else np.exp(-1 * dissimilarity[k][i][j] / percentiles[k][i]) 
                else:
                    compatability[k][i][j] = compatability[2 - k][j][i]

    best_neighbour = np.zeros((4, len(squares)))
    for k in range(4):
        for i in range(len(square)):
            best_neighbour[k][i] = np.argmax(compatability[k][i])
    
    answer = np.zeros(M, N) - 1
    remaining = set(range(M * N))
    answer[M // 2][N // 2] = 0
    remaining.delete(0)

    score = -1
    while True:
        answer = place(answer, compatability, best_neighbour, remaining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str)

    args = parser.parse_args();
    image_name = args.image

    rows = 8
    cols = 8

    solve(image_name, rows, cols)
