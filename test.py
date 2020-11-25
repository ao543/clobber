import numpy as np
from chomp import OnePlane
import matplotlib.pyplot as plt
from PIL import Image as im

X = np.load("features.npy")

Y = np.load("labels.npy")

#decoder = OnePlane.OnePlane(5)

for i in range((X.shape)[0]):
    data = im.fromarray((X[i][0] / 2 * 255).astype(np.uint8))
    label = str(np.argmax(Y[i]))
    str_i = str(i) + '.png'
    data.save('labels/' + label + '/' + str_i)

#print(X.shape)

#print(Y.shape)

#print(X[1])

#label = np.argmax(Y[0])

#print(np.argmax(Y[0]))

#decoder.decode_move(Y[0]).print_mov()

#print(X[0].shape)

#data = im.fromarray((X[1][0]/2 * 255).astype(np.uint8))

#Works
#data.save("test.png")

#data.show()



#gray_skl = X[0]/2

#gray_skl = gray_skl.squeeze()

#print(gray_skl)

#plt.imshow(gray_skl, cmap="gray")

#plt.show()

#print(Y[1])

#move = decoder.decode_move(Y[1])

#move.print_mov()







