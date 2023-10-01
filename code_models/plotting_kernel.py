import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def cauchy_kernel(T, sigma, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.squared_difference(xs_in, xs_out)
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = tf.math.divide(sigma, (distance_matrix_scaled + 1.))

    alpha = 0.001
    eye = tf.eye(num_rows=kernel_matrix.shape.as_list()[-1])
    return kernel_matrix + alpha * eye

kernel = cauchy_kernel(500, 1, 1)

matplotlib.rc('image', cmap='seismic')
plt.imshow(np.log10(kernel),extent=[0,10,10,0])
cbar = plt.colorbar()
cbar.solids.set_edgecolor("face")


print(kernel.numpy())
plt.title ('Prior kernel')
plt.savefig('./figures/covariance_matrix/kernel.eps',format='eps')
plt.savefig('./figures/covariance_matrix/kernel.png')
plt.show()
