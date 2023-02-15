import tensorflow as tf

class GAN:
    def __init__(self):
        # define generator and discriminator networks
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        # define loss functions and optimizers
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def build_generator(self):
        # TODO: define generator network architecture
        pass

    def build_discriminator(self):
        # TODO: define discriminator network architecture
        pass

    def generate_samples(self, num_samples, seed_mp3):
        # TODO: generate new MP3s using the generator network and seed MP3
        pass

    @tf.function
    def train_generator(self, input_noise):
        # TODO: train the generator network
        pass

    @tf.function
    def train_discriminator(self, real_mp3s, fake_mp3s):
        # TODO: train the discriminator network
        pass

    def train(self, dataset, epochs, batch_size, seed_mp3s):
        for epoch in range(epochs):
            for batch in dataset.batch(batch_size):
                # generate random noise for generator input
                noise = tf.random.normal([batch_size, noise_dim])

                # train discriminator on real and fake MP3s
                real_mp3s = batch
                fake_mp3s = self.generator(noise)
                self.train_discriminator(real_mp3s, fake_mp3s)

                # train generator to fool discriminator
                self.train_generator(noise)

            # generate new MP3s using seed MP3s
            if epoch % 100 == 0:
                generated_mp3s = self.generate_samples(num_samples=len(seed_mp3s), seed_mp3=seed_mp3s)
                # TODO: save generated MP3s to disk and display progress
