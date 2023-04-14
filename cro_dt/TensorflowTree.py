import numpy as np
import cro_dt.VectorTree as vt
import tensorflow as tf

@tf.function
def dt_matrix_fit(X, _, W, depth, n_classes, X_, Y_, M):
    N = X.shape[0]

    Z = tf.cast(tf.sign(tf.matmul(W, tf.transpose(X_))), tf.int32)
    Z_ = tf.clip_by_value(tf.subtract(tf.matmul(M, Z), depth - 1), 0, 1)
    R = tf.math.multiply(Z_, Y_)

    count_0s = tf.subtract(N, tf.reduce_sum(Z_, axis=1))

    R_ = tf.cast(R, tf.int32)

    BC = tf.math.bincount(R_, minlength=n_classes, axis=-1)
    C0 = tf.cast(tf.transpose(tf.pad(tf.convert_to_tensor([count_0s]), tf.constant([[0, n_classes - 1, ], [0, 0]]), "CONSTANT")), tf.int32)
    labels = tf.argmax(tf.subtract(BC, C0), axis=1)
    accuracy = tf.divide(tf.reduce_sum(tf.reduce_max(tf.subtract(BC, C0), axis=1)), N)

    return accuracy, labels

def dt_matrix_fit_wrapped(X, _, W, depth, n_classes, X_, Y_, M):
    # with tf.device("/GPU:0"):
    return dt_matrix_fit(X, _, W, depth, n_classes, X_, Y_, M)