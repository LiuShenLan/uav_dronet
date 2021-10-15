import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt
import numpy as np

y = np.array([0.7, 0., 0. , 0.5 , 0.5 , 0.5 ], dtype='float32')
y_pred = np.reshape(y, [-1, 6])
out_mu, out_pi = np.split(y_pred, 2, axis=1)
cat = tfd.Categorical(logits=tf.expand_dims(out_pi,1))
component_splits = [1,1,1]
mus = np.split(out_mu, 3, axis=1)

out_sigma = np.array([[0.1,0.1,0.1]], dtype='float32')
sigs = np.split(out_sigma, 3, axis=1)
coll = [tfd.Laplace(loc=loc, scale=scale) for loc, scale
                in zip(mus, sigs)]
mixture = tfd.Mixture(cat=cat, components=coll)
with tf.Session() as sess:
    x = tf.expand_dims(tf.linspace(-1., 1., int(1e3)),1).eval()
    plt.plot(x, mixture.prob(x).eval());
    plt.savefig("abc.png")
