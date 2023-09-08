import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate(epoch, x_values_list, y_values_list, x_values_list_2, y_values_list_2, scatters):
    plt.clf()
    sc = plt.scatter(x_values_list[epoch], y_values_list[epoch],color='r',label='first')
    sc = plt.scatter(x_values_list_2[epoch], y_values_list_2[epoch],color='g',label='second')
    scatters.append(sc)

if __name__ == '__main__':
    epochs = 10
    fig = plt.figure()
    scatters = []
    x_val_list = []
    y_val_list = []
    x_val_list_2 = []
    y_val_list_2 = []
    
    for i in range(epochs):
        y_val = np.random.randint(-5, 5, 100)
        x_val = np.random.randint(-10, 10, 100)
        y_val_2 = np.random.randint(-5, 5, 100)
        x_val_2 = np.random.randint(-10, 10, 100)
        x_val_list.append(x_val)
        y_val_list.append(y_val)
        x_val_list_2.append(x_val_2)
        y_val_list_2.append(y_val_2)

    ani = FuncAnimation(fig, animate, frames=epochs, fargs=(x_val_list, y_val_list,x_val_list_2, y_val_list_2, scatters), interval=500)
    ani.save('animation.gif', writer='pillow')
    # plt.show()
