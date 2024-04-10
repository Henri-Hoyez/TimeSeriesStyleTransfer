import matplotlib.pyplot as plt
import numpy as np
import io

def plot_to_buff(generations:np.ndarray, nvertical:int=3, nhoriz:int=3):
    legend = [f"feat {j}" for j in range(generations.shape[-1])]

    fig = plt.figure(figsize=(18, 10))
    plt.suptitle("Generations After GAN Training.")

    for i in range(nvertical* nhoriz):
        ax = plt.subplot(nvertical, nhoriz, i+ 1)
        ax.set_title(f"sequence {i+1}")

        plt.plot(generations[i])
        ax.grid(True)
        plt.legend(legend)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf




