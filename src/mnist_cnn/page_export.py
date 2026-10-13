from models.MNIST.page_export import *
from models.MNIST.page_export import _image, _render
import sys
if __name__ == "__main__":
    _render(*sys.argv[1:])
