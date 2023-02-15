import tensorflow as tf

class GAN:
    def __init__(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.losses = {}

    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(100,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Reshape((16, 16, 256)),
            tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ])

        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 1]),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
        ])

        return model

    def generate_samples(self, num_samples, seed_mp3, output_dir, epoch):
        # load seed MP3s and preprocess for generator input
        seed_mp3s = [tf.io.read_file(mp3) for mp3 in seed_mp3]
        seed_mp3s = [tf.audio.decode_mp3(mp3) for mp3 in seed_mp3s]
        seed_mp3s = [tf.audio.encode_wav(tf.expand_dims(mp3.audio, axis=-1), sample_rate=mp3.sample_rate) for mp3 in seed_mp3s]
        seed_mp3s = [tf.audio.decode_wav(mp3) for mp3 in seed_mp3s]
        seed_mp3s = [tf.image.resize(mp3.audio, (128, 128)) for mp3 in seed_mp3s]
        seed_mp3s = [tf.cast(mp3.audio, tf.float32) / 255.0 for mp3 in seed_mp3s]
        seed_mp3s = [tf.expand_dims(mp3, axis=0) for mp3 in seed_mp3s]

        # generate new MP3s using generator network
        generated_mp3s = []
        for i in range(num_samples):
            noise = tf.random.normal([1, 100])
            generated_mp3 = self.generator(noise)
            generated_mp3s.append(generated_mp3)

        # postprocess generated MP3s
        generated_mp3s = [mp3 * 255.0 for mp3 in generated_mp3s]
        generated_mp3s = [tf.cast(mp3, tf.int16) for mp3 in generated_mp3s]
        generated_mp3s = [tf.squeeze(mp3, axis=-1) for mp3 in generated_mp3s]
        generated_mp3s = [tf.audio.encode_wav(mp3, sample_rate=44100) for mp3 in generated_mp3s]

        # save generated MP3s to disk
        for i, mp3 in enumerate(generated_mp3s):
            filename = f'generated_mp3_epoch{epoch}_sample{i}.mp3'
            file_path = f'{output_dir}/{filename}'
            tf.io.write_file(file_path, mp3)

        return generated_mp3s

    @tf.function
    def train_generator(self, input_noise):
        with tf.GradientTape() as tape:
            generated_mp3s = self.generator(input_noise)
            fake_output = self.discriminator(generated_mp3s)
            generator_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)

        gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.losses['generator'] = generator_loss

    @tf.function
    def train_discriminator(self, real_mp3s, fake_mp3s):
        with tf.GradientTape() as tape:
            real_output = self.discriminator(real_mp3s)
            fake_output = self.discriminator(fake_mp3s)

            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss

        gradients = tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        self.losses['discriminator'] = total_loss

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
        # load seed MP3s and preprocess for generator input
        seed_mp3s = [tf.io.read_file(mp3) for mp3 in seed_mp3]
        seed_mp3s = [tf.audio.decode_mp3(mp3) for mp3 in seed_mp3s]
        seed_mp3s = [tf.audio.encode_wav(tf.expand_dims(mp3.audio, axis=-1), sample_rate=mp3.sample_rate) for mp3 in seed_mp3s]
        seed_mp3s = [tf.audio.decode_wav(mp3) for mp3 in seed_mp3s]
        seed_mp3s = [tf.image.resize(mp3.audio, (128, 128)) for mp3 in seed_mp3s]
        seed_mp3s = [tf.cast(mp3.audio, tf.float32) / 255.0 for mp3 in seed_mp3s]
        seed_mp3s = [tf.expand_dims(mp3, axis=0) for mp3 in seed_mp3s]

        # generate new MP3s using generator network
        generated_mp3s = []
        for i in range(num_samples):
            noise = tf.random.normal([1, 100])
            generated_mp3 = self.generator(noise)
            generated_mp3s.append(generated_mp3)

        # postprocess generated MP3s
        generated_mp3s = [mp3 * 255.0 for mp3 in generated_mp3s]
        generated_mp3s = [tf.cast(mp3, tf.int16) for mp3 in generated_mp3s]
        generated_mp3s = [tf.squeeze(mp3, axis=-1) for mp3 in generated_mp3s]
        generated_mp3s = [tf.audio.encode_wav(mp3, sample_rate=44100) for mp3 in generated_mp3s]

        return generated_mp3s
