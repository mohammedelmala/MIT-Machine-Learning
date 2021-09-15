import numpy as np


def perceptron(datax, datay):
    n, d = datax.shape
    theta = np.zeros(d)
    mistakes = 0
    hyperplane = []

    for t in range(10):
        changed = False
        for i in range(n):
            x = datax[i]
            y = datay[i]
            if y * (x@theta) <= 0:
                changed = True
                theta = theta + y*x
                mistakes += 1
                hyperplane.append(theta)

        if not changed:
            break

    return theta, mistakes, hyperplane

# ts = np.array([[-1, -1, 1], [1, 0, -1], [-1, 1.5, 1]])


# ts = np.array([[1, 0, -1], [-1, 1.5, 1], [-1, -1, 1], ])

# ts = np.array([[-1, -1, 1], [1, 0, -1], [-1, 10, 1]])

ts = np.array([[1, 0, -1], [-1, 10, 1], [-1, -1, 1]])

print(ts)

theta, mistakes, hyperplane = perceptron(ts[:, :2], ts[:, 2])
print(f'theta:{theta}')
print(f'mistakes:{mistakes}')
print(f'hyperplane:{hyperplane}')
