import matplotlib.pyplot as plt

def plot(data, title='', xlabel='', ylabel='', filename=None):
    """Simple plot function"""
    plt.figure(figsize=(12, 6))
    plt.plot(data)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    if filename:
        plt.savefig(filename)
    plt.close()
