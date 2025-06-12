# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:14:47 2025

@author: duminil
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tf.keras.models import Model
from tf.keras.layers import Input
from tf.keras.optimizers import Adam
from tf.keras.initializers import RandomNormal
from tf.keras.layers import Conv2D, UpSampling2D, Add, MaxPooling2D, Multiply, LeakyReLU, Activation, Concatenate, Dropout
from loss_utils import FeatureMatchingLoss, VGGFeatureMatchingLoss


class SFTLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(SFTLayer, self).__init__()

    # def build(self, filter_):
        self.cnn = Conv2D(input_dim, (1,1), padding='same')
        
    def call(self, features, cond):
        c1 = self.cnn(cond)
        c2 = self.cnn(c1)
        
        f = tfa.layers.InstanceNormalization(axis=-1)(features)
        lambda_i = Multiply()([c2, f])
        
        c3 = self.cnn(cond)
        c4 = self.cnn(c3)
        
        beta_i = Add()([c4, lambda_i])
        
        return beta_i
    
    
    
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()

    # def build(self, filter_):
        self.cnn = Conv2D(input_dim, (1,1), padding='same')
        
    def call(self, features):
        c0 = features
        c1 = self.cnn(c0)
        softmax = tf.keras.layers.Softmax()(c1)
        
        c2 = Multiply()([softmax, c0])
        c2 = self.cnn(c2)
        c2 = tf.keras.layers.LayerNormalization()(c2)
        c2 = tf.keras.layers.ReLU()(c2)
        c2 = self.cnn(c2)
        
        output = Add()([c0, c2])
        
        return output
    
    
                  
def resnet_block(n_filters, input_layer):

    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first convolutional layer
    g = Conv2D(n_filters, (4,4), padding='same', kernel_initializer=init)(input_layer)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (4,4), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    print("g res block", g.shape)
    print("input_layer", input_layer.shape)
    
    # x_skip = tf.keras.layers.Conv2D(n_filters, (1,1), strides = (2,2))(input_layer)
    g = Concatenate()([g, input_layer])
    
    g = Activation('relu')(g)
 
    return g

def up_res_block(n_filters, input_layer, init):


    g = tfa.layers.InstanceNormalization(axis=-1)(input_layer)
    g = Activation('relu')(g)
    g = Conv2D(n_filters, (4,4), padding='same', kernel_initializer=init)(g)

    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(n_filters, (4,4), padding='same', kernel_initializer=init)(g)
    
    res = tfa.layers.InstanceNormalization(axis=-1)(input_layer)
    res = Activation('relu')(res)
    res = Conv2D(n_filters, (1,1), padding='same', kernel_initializer=init)(res)
    
    # x_skip = tf.keras.layers.Conv2D(n_filters, (1,1), strides = (2,2))(input_layer)
    g = Add()([g, res])

    g = UpSampling2D((2,2))(g)
 
    return g
    

def res_downsample_block(n_filters, input_layer):

    init = RandomNormal(stddev=0.02)
    
    g1 = Conv2D(n_filters,(4,4), padding='same', activation='relu', kernel_initializer=init)(input_layer)
    g = tfa.layers.InstanceNormalization(axis=-1)(g1)
    
    g = Conv2D(n_filters,(4,4), padding='same', activation='relu', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)
    
    res = Conv2D(n_filters,(1,1), padding='same', kernel_initializer=init)(g1)
    res = tfa.layers.InstanceNormalization(axis=-1)(res)
    resconnection = Add()([res, g])
    act = Activation("relu")(resconnection)

    pool2 = MaxPooling2D((2,2))(act)
    # pool2 = Dropout(0.2)(pool2)

    return pool2


    
def define_generator(image_shape):
    
    in_image = Input(shape=image_shape)

# =============================================================================
# Encodeur 
# =============================================================================
    g = resnet_block(64, in_image)
    print("64", g.shape)
    # g = Concatenate()([g, seg1])
    
    
    skip0 = g
    # print('64 shape1', g.shape)
    # d128
    g = res_downsample_block(128, g)
    # g = Concatenate()([g, seg2])

    
    skip1 = g
    # print('downsample 128', g.shape)
    # d256
    g = res_downsample_block(256, g)
    # g = Concatenate()([g, seg3])

    skip2 = g
    print('skip 256', g.shape)

# =============================================================================
# Bottleneck
# =============================================================================
    g = Conv2D(256,(4,4), padding='same', activation='relu')(g)
#------------------------------------------------------------------------------
    
    init = RandomNormal(stddev=0.02)
    
    g = up_res_block(256, g, init)
    print('downsample 256', g.shape)
    g = Add()([g, UpSampling2D((2,2))(skip2)])
    
    Attention = AttentionLayer(256)
    g = Attention(g)
    
    g = up_res_block(128, g, init)
    g = tf.image.resize_with_pad(g, skip1.shape[1], skip1.shape[2])
                                
    print('up res 128', g.shape)
    print('skip1', skip1.shape)
    g = Add()([g, skip1])
    print('concat', g.shape)
    Attention = AttentionLayer(128)
    g = Attention(g)
    
    # x128 = Concatenate(axis=-1)([g, sec_block])
    # print('upsample 128', g.shape)
    
    g = up_res_block(65, g, init)
    g = tf.image.resize_with_pad(g, skip0.shape[1], skip0.shape[2])
    
    g = Add()([g, skip0])
    
    Attention = AttentionLayer(65)
    g = Attention(g)
        
    g = Conv2D(1, (7,7), padding='same', kernel_initializer=init)(g)
    g = tfa.layers.InstanceNormalization(axis=-1)(g)

    out_image = Activation('tanh')(g)

    
    # define model
    model = Model(in_image, out_image)
    model.summary()
 
    return model



# PatchGan 70x70
def define_discriminator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    input_image_A = Input(shape=image_shape)
    

    d = Conv2D(64, (4,4), padding='same', kernel_initializer=init)(input_image_A)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    x1  = LeakyReLU(alpha=0.2)(d)
    x1 = Dropout(0.2)(x1)
    
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x1)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    x2 = LeakyReLU(alpha=0.2)(d)
    x2 = Dropout(0.2)(x2)
    
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x2)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    x3 = LeakyReLU(alpha=0.2)(d)
    x3 = Dropout(0.2)(x3)
    
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x3)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    x4 = LeakyReLU(alpha=0.2)(d)
    x4 = Dropout(0.2)(x4)
    
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(x4)
    d = tfa.layers.InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # patch output
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)

    model = Model(input_image_A, patch_out)
    # compile model
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.000002), loss_weights=[0.5])
    return model

class CycleGan(tf.keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_Y,
        lambda_cycle=15.0,
        lambda_identity=0.5,
    ):
        super().__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.vgg_loss = VGGFeatureMatchingLoss()
        self.feature_matching_loss = FeatureMatchingLoss()


    def train_step(self, batch_data):
        # x is img and y is sketch
        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:
            
            # img to fake sketch
            fake_y = self.gen_G(real_x, training=True)
            cycled_x = self.gen_F(fake_y, training=True)
            
            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)


            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_x, cycled_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(real_x, cycled_x, max_val=2.0))
            ssim_metric = tf.image.ssim(real_x, cycled_x, max_val=2.0)
            
            # vgg_loss = 0.01 * self.vgg_loss(real_y, fake_y)
            # feature_loss  = self.feature_matching_loss(disc_real_y, fake_y)

            total_loss = gen_G_loss + id_loss_G + (self.lambda_cycle * ssim_loss) #+ vgg_loss 

            # Discriminator loss
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        all_trainable_variables = (self.gen_G.trainable_variables + self.gen_F.trainable_variables)
        grads = tape.gradient(total_loss, all_trainable_variables)

        # Get the gradients for the discriminators
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads, all_trainable_variables)
        )

        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss,
            # "F_loss": total_loss_F,
            # "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
            "ssim_metric": ssim_metric

        }
    def call(self, batch_data, training=False): 
       pass