import tensorflow as tf
from model import GAN
from dataset import load_mp3_dataset

def train(data_dir, batch_size, epochs, noise_dim, seed_mp3s, output_dir):
    # set up GPU acceleration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # load MP3 dataset
    dataset = load_mp3_dataset(data_dir, batch_size=batch_size)

    # instantiate GAN model
    gan = GAN()

    # set up loss metrics
    gan.discriminator_loss = tf.keras.metrics.Mean(name='discriminator_loss')
    gan.generator_loss = tf.keras.metrics.Mean(name='generator_loss')

    # train GAN model
    for epoch in range(epochs):
        for batch in dataset:
            # generate input noise for generator
            noise = tf.random.normal([batch_size, noise_dim])

            # train discriminator
            real_mp3s = batch
            fake_mp3s = gan.generator(noise)
            gan.train_discriminator(real_mp3s, fake_mp3s)

            # generate new samples every 100 epochs
            if epoch % 100 == 0:
                generated_mp3s = gan.generate_samples(num_samples=5, seed_mp3=seed_mp3s, output_dir=output_dir, epoch=epoch)

            # train generator
            noise = tf.random.normal([batch_size, noise_dim])
            gan.train_generator(noise)

            # update loss metrics
            gan.discriminator_loss(gan.losses['discriminator'])
            gan.generator_loss(gan.losses['generator'])

        # display training progress every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}')
            print(f'Discriminator Loss: {gan.discriminator_loss.result()}')
            print(f'Generator Loss: {gan.generator_loss.result()}')
            gan.discriminator_loss.reset_states()
            gan.generator_loss.reset_states()
