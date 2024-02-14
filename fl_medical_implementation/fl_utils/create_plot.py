import matplotlib.pyplot as plt

def plot_multiple(y_maps):
        print("In this function")
        for item,value in y_maps.items():
            plt.plot(range(1, len(value) + 1), value, '.-', label=item)
        plt.legend()
        plt.title("Accuracy Curves")
        plt.savefig('accuracy.png')