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

    # compile the model with the Adam optimizer and binary crossentropy loss
    gan.discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    gan.generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    
    # compile the GAN model as a whole, using the discriminator loss and no metrics
    gan.discriminator.trainable = False
    gan.model = tf.keras.Sequential([gan.generator, gan.discriminator])
    gan.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    # set up loss metrics
    discriminator_loss_metric = tf.keras.metrics.Mean(name='discriminator_loss')
    generator_loss_metric = tf.keras.metrics.Mean(name='generator_loss')

    # train GAN model
    for epoch in range(epochs):
        for batch in dataset:
            # generate input noise for generator
            noise = tf.random.normal([batch_size, noise_dim])

            # train discriminator
            real_mp3s = batch
            fake_mp3s = gan.generator(noise)
            discriminator_loss = discriminator_loss_metric(gan.train_discriminator(real_mp3s, fake_mp3s))
            discriminator_loss_metric.update_state(discriminator_loss)

            # generate new samples every 100 epochs
            if epoch % 100 == 0:
                generated_mp3s = gan.generate_samples(num_samples=5, seed_mp3=seed_mp3s, output_dir=output_dir, epoch=epoch)

            # train generator
            noise = tf.random.normal([batch_size, noise_dim])
            generator_loss = generator_loss_metric(gan.train_generator(noise))
            discriminator_loss_metric.update_state(generator_loss)

        # display training progress every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}')
            print(f'Discriminator Loss: {discriminator_loss_metric.result()}')
            print(f'Generator Loss: {generator_loss_metric.result()}')
            discriminator_loss_metric.reset_states()
            generator_loss_metric.reset_states()
