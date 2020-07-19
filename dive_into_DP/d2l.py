from matplotlib import pyplot as plt
from IPython import display
from mxnet import nd

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)):
#    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
def linreg(X, w, b):
    """Linear regression."""
    return nd.dot(X, w) + b

def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2