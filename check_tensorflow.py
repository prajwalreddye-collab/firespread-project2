try:
    import tensorflow as tf

    print("TensorFlow is installed!")
    print("Version:", tf.__version__)

except ImportError:
    print("TensorFlow is not installed.")