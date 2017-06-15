import numpy as np
import tensorflow as tf

def tri(a, m, s, x, t):
    b = 1.0 / s**2.0
    return a * tf.maximum(0.0, m - b * l1_norm(x, t))

def l1_norm(x, t):
    n, d = x.shape
    return tf.norm(x.reshape((n, 1, d)) - t, ord=1, axis=2)

