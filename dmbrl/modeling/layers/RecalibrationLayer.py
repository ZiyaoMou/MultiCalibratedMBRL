import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class RecalibrationLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, emb_dim=None, hidden=16, name='RecalibrationLayer'):
        super().__init__(name=name)
        self.out_dim   = out_dim
        self.emb_dim   = emb_dim 
        self.hidden    = hidden

        # Initialize A to 1 and B to 0 for identity mapping when not calibrated
        self.A = self.add_weight('A', shape=[1, out_dim],
                                 initializer='uniform', trainable=True)
        self.B = self.add_weight('B', shape=[1, out_dim],
                                 initializer='uniform', trainable=True)

        if emb_dim is not None:
            self.fc1 = tf.keras.layers.Dense(hidden, activation='relu')
            self.fc2 = tf.keras.layers.Dense(2 * out_dim, activation=None)

    def get_output_dim(self):
        return self.out_dim

    def call(self, x, domain_vec=None, activation=True):
        x = tf.cast(x, tf.float32)
        if self.emb_dim is None or domain_vec is None:
            A, B = self.A, self.B
        else:
            delta = self.fc2(self.fc1(domain_vec))              
            dA, dB = tf.split(delta, 2, axis=-1)
            A = self.A * (1.0 + tf.tanh(dA))         
            B = self.B + dB

        out = x * A + B
        return tf.nn.sigmoid(out) if activation else out

    def inv_call(self, y, domain_vec=None, activation=True):
        y = tf.cast(y, tf.float32)
        if activation:
            y = tf.log(y / (1. - y))                               # σ⁻¹(y)

        if self.emb_dim is None or domain_vec is None:
            A, B = self.A, self.B
        else:
            delta = self.fc2(self.fc1(domain_vec))
            dA, dB = tf.split(delta, 2, axis=-1)
            A = self.A * (1.0 + tf.tanh(dA))
            B = self.B + dB
        return (y - B) / A
