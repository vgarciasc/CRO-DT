import sys
sys.path.append(".")

import numpy as np
import cro_dt.VectorTree as vt
import tensorflow as tf

@tf.function
def dt_matrix_fit(X, _, W, depth, n_classes, X_, Y_, M, N):
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

@tf.function
def dt_matrix_fit_nb(X, _, W, depth, n_classes, X_, Y_, M, N, n_leaves):
    Z = tf.cast(tf.sign(tf.matmul(W, tf.transpose(X_))), tf.int32)
    Q = tf.clip_by_value(tf.subtract(tf.matmul(M, Z), depth - 1), 0, 1)
    Y_Q = tf.cast(tf.math.multiply(Y_, Q), tf.int32)
    R = tf.math.bincount(Y_Q, axis=-1)
    R_ = tf.slice(R, [0, 1], [n_leaves, n_classes])

    labels = tf.argmax(R_, axis=1)
    accuracy = tf.divide(tf.reduce_sum(tf.reduce_max(R_, axis=1)), N)

    return accuracy, labels

@tf.function
def dt_matrix_fit_batch(X, _, W_total, depth, n_classes, X_, Y_, M, N, n_leaves, batch_size):
    Z = tf.cast(tf.sign(tf.matmul(W_total, tf.transpose(X_))), tf.int32)
    Z_ = tf.cast(tf.clip_by_value(tf.subtract(tf.matmul(M, Z), depth - 1), 0, 1), tf.int32)
    R_ = tf.math.multiply(Z_, Y_)

    count_0s = tf.subtract(N, tf.reduce_sum(Z_, axis=2))

    R_2 = tf.reshape(R_, (batch_size * n_leaves, N))
    BC2 = tf.math.bincount(R_2, axis=-1)
    BC = tf.reshape(BC2, [batch_size, n_leaves, n_classes])

    C0 = tf.cast(tf.transpose(tf.pad(tf.convert_to_tensor([count_0s]), tf.constant([[0, n_classes - 1, ], [0, 0], [0, 0]]), "CONSTANT"), (1, 2, 0)), tf.int32)
    labels = tf.argmax(tf.subtract(BC, C0), axis=2)
    accuracy = tf.divide(tf.reduce_sum(tf.reduce_max(tf.subtract(BC, C0), axis=2), axis=1), N)

    return accuracy, labels

@tf.function
def dt_matrix_fit_batch_nb(X, _, W_total, depth, n_classes, X_, Y_, M, N, n_leaves, batch_size):
    Z = tf.cast(tf.sign(tf.matmul(W_total, tf.transpose(X_))), tf.int32)
    Q = tf.cast(tf.clip_by_value(tf.subtract(tf.matmul(M, Z), depth - 1), 0, 1), tf.int32)
    Y_Q = tf.math.multiply(Y_, Q)

    R1 = tf.reshape(Y_Q, (batch_size * n_leaves, N))
    R2 = tf.math.bincount(R1, axis=-1)
    R3 = tf.slice(R2, [0, 1], [batch_size * n_leaves, n_classes])
    R = tf.reshape(R3, [Z.shape[0], n_leaves, n_classes])

    labels = tf.argmax(R, axis=2)
    accuracy = tf.divide(tf.reduce_sum(tf.reduce_max(R, axis=2), axis=1), N)

    return accuracy, labels

def dt_matrix_fit_batch_univariate_nb(solutions, X_train_, Y_train_, depth, n_classes, M, N, batch_size):
    W_batch = tf.reshape(tf.convert_to_tensor(solutions, dtype=tf.float64), (len(solutions), 2 ** depth - 1, -1))

    W_thresholds = tf.expand_dims(tf.gather(W_batch, 0, axis=2), 2)
    W_attrs = tf.slice(W_batch, [0, 0, 1], [W_batch.shape[0], W_batch.shape[1], W_batch.shape[2] - 1])
    W_attrs_abs = tf.abs(W_attrs)
    W_attrs_max_mask = tf.equal(W_attrs_abs, tf.expand_dims(tf.reduce_max(W_attrs_abs, axis=2), 2))
    W_univ_batch = tf.concat([W_thresholds, tf.where(W_attrs_max_mask, W_attrs, 0)], axis=2)

    accuracies, _ = dt_matrix_fit_batch_nb(None, None, W_univ_batch, depth, n_classes,
                                           X_train_, Y_train_, M, N, 2 ** depth, batch_size)
    return accuracies

def dt_matrix_fit_batch_multivariate_nb(solutions, X_train_, Y_train_, depth, n_classes, M, N, batch_size):
    W_batch = tf.reshape(tf.convert_to_tensor(solutions, dtype=tf.float64), (len(solutions), 2 ** depth - 1, -1))

    accuracies, _ = dt_matrix_fit_batch_nb(None, None, W_batch, depth, n_classes,
                                           X_train_, Y_train_, M, N, 2 ** depth, batch_size)
    return accuracies