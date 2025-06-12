# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:32:25 2025

@author: duminil
"""
import argparse
import tensorflow as tf
from models import define_generator, define_discriminator, CycleGan
from loss_utils import generator_loss_fn, discriminator_loss_fn
from data import ImageGenerator

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_shape", default = (256,256,1))
    parser.add_argument("--path_to_csv", default = "E:/flowers.csv")
    parser.add_argument("--checkpoint_filepath", default = "D:/0-Works/models/test_2.{epoch:03d}")
    parser.add_argument("--batch_size", default = 1)
    parser.add_argument("--epochs", default = 150)
    return parser.parse_args()

def main():
    args = parse_arguments()
    gen_G = define_generator(args.image_shape)
    gen_F = define_generator(args.image_shape)
    disc_Y = define_discriminator(args.image_shape)
    
    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_Y=disc_Y
    )
    
    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
        disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )
    

    vgg_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False)
    train_gen = ImageGenerator(args.path_to_csv, args.batch_size, vgg_model, (args.batch_size, args.image_shape), transformation=False)
    
    # cycle_gan_model.load_weights('D:/0-Works/models/test.100')
    
    model_checkpoint_callback =[
        tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_filepath, save_weights_only=True)
    ]
    
    cycle_gan_model.fit(
        train_gen,
        # validation_split=0.2
        epochs=args.epochs,
        callbacks=[model_checkpoint_callback],
    )

if __name__ == "__main__":
    main()