import numpy as np
from chomp import OnePlane

X = np.load("features.npy")

Y = np.load("labels.npy")

decoder = OnePlane.OnePlane(5)

print(X[2])

print(Y[1])

move = decoder.decode_move(Y[1])

move.print_mov()







