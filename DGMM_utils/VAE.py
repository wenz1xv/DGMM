import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model
from .dataprocess import alphabet, chars

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z

class VAE(Model):
    def __init__(self, maxlen, batch_size=16, latent_dim=80, ak=1e-2, lr=1e-3):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.onehot_dim = len(alphabet)
        self.alphabet = alphabet
        self.scf = np.where([x.find('Ring') > -1 or x.find('Branch') > -1 for x in alphabet])[0]
        self.maxlen = maxlen
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.train_acc = keras.metrics.SparseCategoricalAccuracy()
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.fp_loss = keras.metrics.Mean(name='fp_loss')
        self.kl_loss = keras.metrics.Mean(name="kl_loss")
        self.total_loss = keras.metrics.Mean(name="total_loss")
        self.ak = ak
        self.lr = lr
        
    def pro2sfi(self, data):
        res = []
        for s in data:
            sfi = ''
            for j in s:
                idx = np.random.choice(np.arange(len(self.alphabet)), p=j)
                token = self.alphabet[idx]
                if self.alphabet[idx].find("[nop]") > -1:
                    break
                sfi+=token
            res.append(sfi)
        return res
        
    def call(self, inp):
        z_mean, z_log_var, z = self.encoder(inp)
        out, fp_pred = self.decoder(z)
        return out
    
    def save(self, path, **kwargs):
        self.encoder.save(path.replace('.keras','')+'_encoder.keras', **kwargs)
        self.decoder.save(path.replace('.keras','')+'_decoder.keras', **kwargs)
    
    def transconv(self, inp, size):
        up = layers.UpSampling1D(size=2)(inp)
        net = layers.Conv1D(size, 2, activation='relu', padding='same')(up)
        return net

    def compile(self, optimizer='none', *args, **kwargs):
        if optimizer =='none':
            self.optimizer = keras.optimizers.SGD(learning_rate=self.lr)
        else:
            self.optimizer = optimizer
        self.encoder.compile(optimizer=self.optimizer)
        self.decoder.compile(optimizer=self.optimizer)
        super(VAE, self).compile(*args, **kwargs)
    
    def get_encoder(self):
        inp_smi = layers.Input(shape=(self.maxlen,), dtype=np.float32)
        net = layers.Embedding(self.onehot_dim, 32, input_length=self.maxlen)(inp_smi)
        conv = layers.GRU(64, return_sequences=True)(net)
        pool = layers.MaxPooling1D(pool_size=2)(conv)
        conv = layers.GRU(128, return_sequences=True)(pool)
        pool = layers.MaxPooling1D(pool_size=2)(conv)
        conv = layers.Conv1D(256, 2, padding='same', activation='relu')(pool)
        net = layers.MaxPooling1D(pool_size=2)(conv)
        net = layers.Flatten()(net)
        net = layers.Dense(256, activation='relu')(net)
        latent_space = layers.Dense(128, activation='relu')(net)
        latent_space = layers.BatchNormalization()(latent_space)

        z_mean = layers.Dense(self.latent_dim,
                              kernel_initializer=keras.initializers.Zeros(),
                              bias_initializer=keras.initializers.Zeros(),
                              name="z_mean")(latent_space)
        z_log_var = layers.Dense(self.latent_dim,
                              kernel_initializer=keras.initializers.Zeros(),
                              bias_initializer=keras.initializers.Zeros(),
                                 name="z_log_var")(latent_space)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(inp_smi, [z_mean, z_log_var, z], name="encoder")
    
    def get_decoder(self):
        input_1 = layers.Input(shape=(self.latent_dim,), dtype=np.float32)
        fp_pred = layers.Dense(167, activation='sigmoid')(input_1)
        net = layers.Reshape((5,-1))(input_1)
        z_mean = net
        de1 = layers.GRU(64, return_sequences=True)(net)
        up = layers.UpSampling1D(size=2)(de1)
        
        z_mean = self.transconv(z_mean, 64)
        up = z_mean + up
        de2 = layers.Dense(128, activation='relu')(up)
        up = layers.UpSampling1D(size=2)(de2)
        z_mean = self.transconv(z_mean, 128)
        up = z_mean + up
        
        de3 = layers.GRU(256, return_sequences=True)(up)
        up = layers.UpSampling1D(size=2)(de3)
        z_mean = self.transconv(z_mean, 256)
        up = z_mean + up
        
        de4 = layers.Dense(64, activation='relu')(up)
        up = layers.UpSampling1D(size=2)(de4)
        net = layers.GRU(64, return_sequences=True)(up)
        
        net = layers.Dense(80, activation='relu')(net)
        net = layers.Dense(80, activation='relu')(net)
        
        out_hot = layers.Dense(self.onehot_dim, activation='softmax')(net)
        return Model(inputs=input_1, outputs=[out_hot, fp_pred], name='decoder')
    
    @property
    def metrics(self):
        return [
            self.train_loss,
            self.fp_loss,
            self.kl_loss,
            self.train_acc,
        ]
    
    def train_step(self, data):
        x = data[0]
        fp = data[1]
#         loss_fn = keras.losses.CategoricalCrossentropy()
        loss_fn1 = keras.losses.BinaryCrossentropy()
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            y_pred, fp_pred = self.decoder(z)
            y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0-1e-5)

            mask = tf.math.logical_not(tf.math.equal(x,127))
            reconstruction_loss = keras.losses.sparse_categorical_crossentropy(x, y_pred)
            reconstruction_loss = tf.boolean_mask(reconstruction_loss, mask)
            reconstruction_loss = 0.8*tf.reduce_mean(reconstruction_loss)
            reconstruction_loss += 0.2*tf.reduce_mean(loss_fn(x, y_pred))
            
            fp_loss = tf.reduce_mean(loss_fn1(fp, fp_pred))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = fp_loss + reconstruction_loss + kl_loss*self.ak
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss.update_state(total_loss)
        self.fp_loss.update_state(fp_loss)
        self.kl_loss.update_state(kl_loss)
        self.train_loss.update_state(reconstruction_loss)
        self.train_acc.update_state(x, y_pred)
        return {
            'loss': self.train_loss.result(),
            'acc': self.train_acc.result(),
            'kl_loss':self.kl_loss.result(),
            'fp_loss':self.fp_loss.result(),
            'total_loss':self.total_loss.result()
            }
    
    def test_step(self, data):
        x = data[0]
        fp = data[1]
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        loss_fn1 = keras.losses.BinaryCrossentropy()
        z_mean, z_log_var, z = self.encoder(x)
        y_pred, fp_pred = self.decoder(z)
        loss = tf.reduce_mean(tf.reduce_sum(keras.losses.sparse_categorical_crossentropy(x, y_pred), axis=1))
        fp_loss = tf.reduce_mean(loss_fn1(fp, fp_pred))
        # Update the metrics.
        self.train_loss.update_state(loss)
        self.fp_loss.update_state(fp_loss)
        self.train_acc.update_state(x, y_pred)
        return {
            'loss': self.train_loss.result(),
            'fp_loss': self.fp_loss.result(),
            'acc': self.train_acc.result()
        }
    
def generateTrainDecoder(decoder_path, train_path, latent_dim=80):
    def transconv(inp, size):
        up = layers.UpSampling1D(size=2)(inp)
        net = layers.Conv1D(size, 2, activation='relu', padding='same')(up)
        return net
    input_1 = layers.Input(shape=(latent_dim,), dtype=np.float32)
    fp_pred = layers.Dense(167, activation='sigmoid')(input_1)
    net = layers.Reshape((5,-1))(input_1)
    z_mean = net
    de1 = layers.GRU(64, return_sequences=True)(net)
    up = layers.UpSampling1D(size=2)(de1)

    z_mean = transconv(z_mean, 64)
    up = z_mean + up
    de2 = layers.Dense(128, activation='relu')(up)
    up = layers.UpSampling1D(size=2)(de2)
    z_mean = transconv(z_mean, 128)
    up = z_mean + up

    de3 = layers.GRU(256, return_sequences=True)(up)
    up = layers.UpSampling1D(size=2)(de3)
    z_mean = transconv(z_mean, 256)
    up = z_mean + up

    de4 = layers.Dense(64, activation='relu')(up)
    up = layers.UpSampling1D(size=2)(de4)
    net = layers.GRU(64, return_sequences=True)(up)

    net = layers.Dense(80, activation='relu')(net)
    net = layers.Dense(80, activation='relu')(net)

    out_hot = layers.Dense(len(alphabet), activation='relu')(net)
    train_decoder = Model(inputs=input_1, outputs=[out_hot, fp_pred], name='decoder')
    critic_decoder = load_model(decoder_path)
    
    for idx, l in enumerate(train_decoder.layers[1:]):
        l.set_weights(critic_decoder.layers[idx+1].get_weights())
    optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=.6)
    train_decoder.compile(optimizer=optimizer)
    train_decoder.save(train_path)

class VAEsmi(Model):
    def __init__(self, maxlen, batch_size=16, latent_dim=80, ak=1e-2, lr=1e-3):
        super(VAEsmi, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.onehot_dim = len(chars)+2
        self.maxlen = maxlen
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.train_acc = keras.metrics.SparseCategoricalAccuracy()
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.fp_loss = keras.metrics.Mean(name='fp_loss')
        self.kl_loss = keras.metrics.Mean(name="kl_loss")
        self.total_loss = keras.metrics.Mean(name="total_loss")
        self.ak = ak
        self.lr = lr
        
    def call(self, inp):
        z_mean, z_log_var, z = self.encoder(inp)
        out, fp_pred = self.decoder(z)
        return out
    
    def save(self, path, **kwargs):
        self.encoder.save(path.replace('.keras','')+'_encoder.keras', **kwargs)
        self.decoder.save(path.replace('.keras','')+'_decoder.keras', **kwargs)
    
    def transconv(self, inp, size):
        up = layers.UpSampling1D(size=2)(inp)
        net = layers.Conv1D(size, 2, activation='relu', padding='same')(up)
        return net

    def compile(self, optimizer='none', *args, **kwargs):
        if optimizer =='none':
            self.optimizer = keras.optimizers.SGD(learning_rate=self.lr)
        else:
            self.optimizer = optimizer
        self.encoder.compile(optimizer=self.optimizer)
        self.decoder.compile(optimizer=self.optimizer)
        super(VAEsmi, self).compile(*args, **kwargs)
    
    def get_encoder(self):
        inp_smi = layers.Input(shape=(self.maxlen,), dtype=np.float32)
        net = layers.Embedding(self.onehot_dim, 32, input_length=self.maxlen)(inp_smi)
        conv = layers.GRU(64, return_sequences=True)(net)
        pool = layers.MaxPooling1D(pool_size=2)(conv)
        conv = layers.GRU(128, return_sequences=True)(pool)
        pool = layers.MaxPooling1D(pool_size=2)(conv)
        conv = layers.Conv1D(256, 2, padding='same', activation='relu')(pool)
        net = layers.MaxPooling1D(pool_size=2)(conv)
        net = layers.Flatten()(net)
        net = layers.Dense(256, activation='relu')(net)
        latent_space = layers.Dense(128, activation='relu')(net)
        latent_space = layers.BatchNormalization()(latent_space)

        z_mean = layers.Dense(self.latent_dim,
                              kernel_initializer=keras.initializers.Zeros(),
                              bias_initializer=keras.initializers.Zeros(),
                              name="z_mean")(latent_space)
        z_log_var = layers.Dense(self.latent_dim,
                              kernel_initializer=keras.initializers.Zeros(),
                              bias_initializer=keras.initializers.Zeros(),
                                 name="z_log_var")(latent_space)
        z = Sampling()([z_mean, z_log_var])
        return keras.Model(inp_smi, [z_mean, z_log_var, z], name="encoder")
    
    def get_decoder(self):
        input_1 = layers.Input(shape=(self.latent_dim,), dtype=np.float32)
        fp_pred = layers.Dense(167, activation='sigmoid')(input_1)
        net = layers.Reshape((5,-1))(input_1)
        z_mean = net
        de1 = layers.GRU(64, return_sequences=True)(net)
        up = layers.UpSampling1D(size=2)(de1)
        
        z_mean = self.transconv(z_mean, 64)
        up = z_mean + up
        de2 = layers.Dense(128, activation='relu')(up)
        up = layers.UpSampling1D(size=2)(de2)
        z_mean = self.transconv(z_mean, 128)
        up = z_mean + up
        
        de3 = layers.GRU(256, return_sequences=True)(up)
        up = layers.UpSampling1D(size=2)(de3)
        z_mean = self.transconv(z_mean, 256)
        up = z_mean + up
        
        de4 = layers.Dense(64, activation='relu')(up)
        up = layers.UpSampling1D(size=2)(de4)
        net = layers.GRU(64, return_sequences=True)(up)
        
        net = layers.Dense(80, activation='relu')(net)
        net = layers.Dense(80, activation='relu')(net)
        
        out_hot = layers.Dense(self.onehot_dim, activation='softmax')(net)
        return Model(inputs=input_1, outputs=[out_hot, fp_pred], name='decoder')
    
    @property
    def metrics(self):
        return [
            self.train_loss,
            self.fp_loss,
            self.kl_loss,
            self.train_acc,
        ]
    
    def train_step(self, data):
        x = data[0]
        fp = data[1]
        loss_fn1 = keras.losses.BinaryCrossentropy()
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            y_pred, fp_pred = self.decoder(z)
            y_pred = tf.clip_by_value(y_pred, 1e-5, 1.0-1e-5)

            mask = tf.math.logical_not(tf.math.equal(x,127))
            reconstruction_loss = keras.losses.sparse_categorical_crossentropy(x, y_pred)
            reconstruction_loss = tf.boolean_mask(reconstruction_loss, mask)
            reconstruction_loss = 0.8*tf.reduce_mean(reconstruction_loss)
            reconstruction_loss += 0.2*tf.reduce_mean(loss_fn(x, y_pred))
            
            fp_loss = tf.reduce_mean(loss_fn1(fp, fp_pred))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = fp_loss + reconstruction_loss + kl_loss*self.ak
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss.update_state(total_loss)
        self.fp_loss.update_state(fp_loss)
        self.kl_loss.update_state(kl_loss)
        self.train_loss.update_state(reconstruction_loss)
        self.train_acc.update_state(x, y_pred)
        return {
            'loss': self.train_loss.result(),
            'acc': self.train_acc.result(),
            'kl_loss':self.kl_loss.result(),
            'fp_loss':self.fp_loss.result(),
            'total_loss':self.total_loss.result()
            }
    
    def test_step(self, data):
        x = data[0]
        fp = data[1]
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        loss_fn1 = keras.losses.BinaryCrossentropy()
        z_mean, z_log_var, z = self.encoder(x)
        y_pred, fp_pred = self.decoder(z)
        loss = tf.reduce_mean(tf.reduce_sum(keras.losses.sparse_categorical_crossentropy(x, y_pred), axis=1))
        fp_loss = tf.reduce_mean(loss_fn1(fp, fp_pred))
        # Update the metrics.
        self.train_loss.update_state(loss)
        self.fp_loss.update_state(fp_loss)
        self.train_acc.update_state(x, y_pred)
        return {
            'loss': self.train_loss.result(),
            'fp_loss': self.fp_loss.result(),
            'acc': self.train_acc.result()
        }