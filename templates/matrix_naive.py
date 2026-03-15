result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(3):
    for j in range(3):
        for k in range(3):
            result[i][j] = result[i][j] + A[i][k] * B[k][j]
return result
