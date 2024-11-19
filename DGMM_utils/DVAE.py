# Dencoder
import numpy as np
import tensorflow as tf
import selfies as sf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.models import load_model
from .model_layers import Sampling, VectorQuantizer
from .dataprocess import alphabet

class DVAEencoder(Model):
    def __init__(self, model_path,maxlen=80, model_name='test', lr=1e-3, 
                 ema=True, gene_range=8, gene_length=320, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.onehot_dim = len(alphabet)
        self.maxlen = maxlen
        self.num_embeddings = gene_range
        self.model_name = model_name
        self.ema = ema
        self.gene_length = gene_length
        self.model_path = model_path
        
        self.init_model()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")
        self.perplexity_tracker = keras.metrics.Mean(name='perplexity_loss')
        self.latent_mae_tracker = keras.metrics.MeanSquaredError()
        self.student_acc_tracker = keras.metrics.SparseCategoricalAccuracy()
        self.teacher_acc_tracker = keras.metrics.SparseCategoricalAccuracy()
    
        self.lr = lr
        
    def call(self, inputs):
        pass
    def load(self, path, **kwargs):
        pass
        
    def save(self, path, **kwargs):
        self.latent2vec.save(self.model_path +path+f'/latent2vec.h5', **kwargs)
        self.vec2latent.save(self.model_path +path+f'/vec2latent.h5', **kwargs)

    def compile(self, optimizer='none', *args, **kwargs):
        if optimizer =='none':
            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=.6)
        else:
            self.optimizer = optimizer
        
        self.latent2vec.compile(optimizer=self.optimizer)
        self.vec2latent.compile(optimizer=self.optimizer)
        super().compile(*args, **kwargs)
        
    def init_model(self):
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.latent_dim, name="vector_quantizer", ema=self.ema)
        critic_encoder_path = self.model_path +'TVAE/DGMM_sfi_encoder.h5'
        self.critic_encoder = load_model(critic_encoder_path, custom_objects={'Sampling':Sampling})
        self.critic_encoder.trainable = False
        critic_decoder_path = self.model_path +'TVAE/DGMM_sfi_decoder.h5'
        self.critic_decoder = load_model(critic_decoder_path)
        self.critic_decoder.trainable = False
        self.latent2vec = self.get_latent2vec()
        self.vec2latent = self.get_vec2latent()
        
    def get_latent2vec(self):
        inp_smi = layers.Input(shape=(self.maxlen), dtype=np.float32)
        net = layers.RepeatVector(self.latent_dim)(inp_smi)
        net = layers.GRU(64, return_sequences=True)(net)
        net = layers.Dense(512, activation='relu')(net)
        net = layers.Dense(512, activation='relu')(net)
        net = layers.Dropout(0.1)(net)
        net = layers.Dense(512, activation='relu')(net)
        net = layers.GRU(64, return_sequences=True)(net)
        net = layers.Dense(self.gene_length, activation='relu',name='enoutput')(net)
        quantized_latents, perplexity= self.vq_layer(net)
        latent2vec = keras.Model(inp_smi, [quantized_latents, perplexity], name="latent2vec")
        return latent2vec
    
    def get_vec2latent(self):
        inp = layers.Input(shape=(self.latent_dim, self.gene_length), dtype=np.float32)
        net = layers.Dense(self.gene_length, activation='relu')(inp)
        net = layers.GRU(64, return_sequences=True)(net)
        net = layers.Flatten()(net)
        net = layers.Dense(512)(net)
        net = layers.Dense(512)(net)
        net = layers.Dropout(0.1)(net)
        latent = layers.Dense(80)(net)
        vec2latent = keras.Model(inp, latent, name="vec2latent")
        return vec2latent
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.vq_loss_tracker,
            self.perplexity_tracker,
            self.latent_mae_tracker,
            self.student_acc_tracker,
        ]
    
    def perturb_z(self, z, noise_norm=.5):
        if noise_norm > 0.0:
            noise_vec = tf.keras.backend.random_normal(shape=tf.shape(z))
            noise_vec = noise_vec / tf.linalg.norm(noise_vec)
            noise_amp = tf.keras.backend.random_uniform([tf.shape(z)[0], 1], 0, noise_norm)
            return z + (noise_amp * noise_vec)
        else:
            return z
    
    def train_step(self, data):
        inputs = data
        bc_fn = keras.losses.BinaryCrossentropy()
        scc_fn = keras.losses.SparseCategoricalCrossentropy()
        
        kl_loss_fn = keras.losses.KLDivergence()
        
        with tf.GradientTape() as tape:
            gene_mean, gene_var, gene = self.critic_encoder(inputs)
            quantized_latents, perplexity = self.latent2vec(gene)
            gene_pred = self.vec2latent(quantized_latents)
            y_pred, student_fp_pred = self.critic_decoder(gene_pred)
            latent_loss = tf.reduce_mean(tf.reduce_sum((gene - gene_pred) ** 2, axis=-1))
            vq_loss = tf.reduce_sum(self.vq_layer.losses)
            total_loss = latent_loss  + vq_loss
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.perplexity_tracker.update_state(perplexity)
        self.vq_loss_tracker.update_state(vq_loss)
        self.latent_mae_tracker.update_state(gene, gene_pred)
        self.student_acc_tracker.update_state(inputs, y_pred)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
            "perplexity": self.perplexity_tracker.result(),
            'latent_mae': self.latent_mae_tracker.result(),
            'student_acc': self.student_acc_tracker.result(),
            }
    
    def test_step(self, data):
        inputs = data
        bc_fn = keras.losses.BinaryCrossentropy()
        scc_fn = keras.losses.SparseCategoricalCrossentropy()
        gene_mean, gene_var, gene = self.critic_encoder(inputs)
        quantized_latents, perplexity = self.latent2vec(gene)
        gene_pred = self.vec2latent(quantized_latents)
        y_pred, student_fp_pred = self.critic_decoder(gene_pred)

        self.student_acc_tracker.update_state(inputs, y_pred)
        return {
            'accuracy': self.student_acc_tracker.result()
        }


class DVAEdecoder(Model):
    def __init__(self, model_path, maxlen=80, lr=1e-3, 
        gene_range=8, gene_length=320, latent_dim=8, T=3.,alpha=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.onehot_dim = len(alphabet)
        self.maxlen = maxlen
        self.num_embeddings = gene_range
        self.gene_length = gene_length
        self.model_path = model_path
        
        self.init_model()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.student_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.distill_loss_tracker = keras.metrics.Mean(name='latent_loss')
        self.perplexity_tracker = keras.metrics.Mean(name='perplexity_loss')
        self.latent_mae_tracker = keras.metrics.MeanSquaredError()
        self.student_acc_tracker = keras.metrics.SparseCategoricalAccuracy()
        self.teacher_acc_tracker = keras.metrics.SparseCategoricalAccuracy()
    
        self.temperature = T
        self.alpha = alpha
        self.lr = lr
        
    def pro2smi(self, data):
        res = []
        for s in data:
            sfi = ''
            for j in s:
                idx = np.random.choice(np.arange(len(alphabet)), p=j)
                token = alphabet[idx]
                if alphabet[idx].find("[nop]") > -1:
                    break
                sfi+=token
            res.append(sf.decoder(sfi))
        return res


    def qlatent2indices(self, qlatent):
        codebook = self.vq_layer.embeddings
        flattened_inputs = tf.reshape(qlatent, [-1, tf.shape(codebook)[0]])
        similarity = tf.matmul(flattened_inputs, codebook)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum( codebook ** 2, axis=0)
            - 2 * similarity
        )
        return tf.reshape(tf.argmin(distances, axis=1), [tf.shape(qlatent)[0],-1]).numpy()

    def indices2qlatent(self, indices):
        codebook = self.vq_layer.embeddings
        encodings = tf.one_hot(indices, codebook.shape[1])
        latents = tf.matmul(encodings, codebook, transpose_b=True)
        qlatent = tf.reshape(latents, [tf.shape(indices)[0],  codebook.shape[0], -1])
        return qlatent


    def decode_gene(self, gene):
        latent = self.indices2qlatent(genes)
        pros = self.decoder.predict(latent)
        smis = [sf.decoder(pro2sfi(pro)) for pro in pros]
        return smis
        
    def call(self, inputs):
        gene_mean, gene_var, gene = self.critic_encoder(inputs)
        quantized_latents, perplexity = self.latent2vec(gene)
        encoding_indices = self.qlatent2indices(quantized_latents)
        student_out, gene_pred = self.decoder(quantized_latents)
        teacher_out, teacher_fp_pred = self.critic_decoder_train(gene)
        student_pred = tf.nn.softmax(student_out)
        teacher_pred = tf.nn.softmax(teacher_out)
        student_smiles_pred = self.pro2smi(student_pred)
        teacher_smiles_pred = self.pro2smi(teacher_pred)

        res = {
            'teacher_genes': gene.numpy(),
            'teacher_smiles': teacher_smiles_pred,
            'student_smiles': student_smiles_pred,
            'student_latent':quantized_latents.numpy(),
            'indices': encoding_indices.numpy(),
            'perplexity': perplexity
        }
        return res
        
    def save(self, path, **kwargs):
        self.decoder.save(self.model_path + path +f'/Qdecoder.h5', **kwargs)
        
    def load(self, path, **kwargs):
        pass
    
    def compile(self, optimizer='none', *args, **kwargs):
        if optimizer =='none':
            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=.6)
        else:
            self.optimizer = optimizer
        self.decoder.compile(optimizer=self.optimizer)
        super().compile(*args, **kwargs)
        
    def init_model(self):
        self.latent2vec = load_model(self.model_path + 'DVAE/latent2vec.h5', custom_objects={'VectorQuantizer':VectorQuantizer})
        self.codebook = self.latent2vec.layers[-1].embeddings
        self.latent2vec.trainable = False
        self.decoder = self.get_decoder()
        critic_encoder_path = self.model_path + 'TVAE/DGMM_sfi_encoder.h5'
        self.critic_encoder = load_model(critic_encoder_path, custom_objects={'Sampling':Sampling})
        self.critic_encoder.trainable = False
        critic_decoder_path = self.model_path + 'TVAE/DGMM_sfi_decoder_nosoftmax.h5'
        self.critic_decoder_train = load_model(critic_decoder_path)
        self.critic_decoder_train.trainable = False
        critic_decoder_path = self.model_path + 'TVAE/DGMM_sfi_decoder.h5'
        self.critic_decoder = load_model(critic_decoder_path)
        self.critic_decoder.trainable = False
    
    def transconv(self, inp, size):
        up = layers.UpSampling1D(size=2)(inp)
        net = layers.Conv1D(size, 2, activation='relu', padding='same')(up)
        return net
    
    def get_decoder(self):
        inp = layers.Input(shape=(self.latent_dim, self.gene_length), dtype=np.float32)
        net = layers.Dense(self.gene_length, activation='relu')(inp)
        net = layers.Flatten()(net)
        net = layers.Dense(512)(net)
        net = layers.Dropout(0.1)(net)
        input_1 = layers.Dense(80)(net)

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
        
        out_hot = layers.Dense(self.onehot_dim, activation='relu')(net)
        return Model(inputs=inp, outputs=[out_hot, input_1], name='decoder')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.distill_loss_tracker,
            self.student_loss_tracker,
            self.perplexity_tracker,
            self.latent_mae_tracker,
            self.student_acc_tracker,
        ]
    
    def train_step(self, data):
        inputs = data
        scc_fn = keras.losses.SparseCategoricalCrossentropy()
        kl_loss_fn = keras.losses.KLDivergence()
        
        with tf.GradientTape() as tape:
            gene_mean, gene_var, gene = self.critic_encoder(inputs)
            quantized_latents, perplexity = self.latent2vec(gene)
            student_out, gene_pred = self.decoder(quantized_latents)
            teacher_out, teacher_fp_pred = self.critic_decoder_train(gene)
            student_pred = tf.nn.softmax(student_out)
            teacher_pred = tf.nn.softmax(teacher_out)
        
            latent_loss = tf.reduce_mean((gene - gene_pred) ** 2)

            distill_loss = kl_loss_fn(
                tf.nn.softmax(teacher_out / self.temperature),
                tf.nn.softmax(student_out / self.temperature),
            ) * (self.temperature**2)
            
            
            student_loss = tf.reduce_mean(scc_fn(inputs, student_pred))
            
            total_loss = self.alpha * (student_loss) + (1 - self.alpha) * distill_loss
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.distill_loss_tracker.update_state(distill_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.perplexity_tracker.update_state(perplexity)
        self.latent_mae_tracker.update_state(gene, gene_pred)
        self.student_acc_tracker.update_state(inputs, student_pred)
        self.teacher_acc_tracker.update_state(inputs, teacher_pred)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "distill_loss": self.distill_loss_tracker.result(),
            "perplexity": self.perplexity_tracker.result(),
            'latent_mae': self.latent_mae_tracker.result(),
            'student_acc': self.student_acc_tracker.result(),
            'teacher_acc': self.teacher_acc_tracker.result(),
            }
    
    def test_step(self, data):
        inputs = data
        gene_mean, gene_var, gene = self.critic_encoder(inputs)
        quantized_latents, perplexity = self.latent2vec(gene)
        student_out, gene_pred = self.decoder(quantized_latents)
        y_pred = tf.nn.softmax(student_out)
        self.student_acc_tracker.update_state(inputs, y_pred)
        return {
            'accuracy': self.student_acc_tracker.result()
        }
