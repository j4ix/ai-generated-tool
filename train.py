import tensorflow as tf
from model import GAN
from dataset import load_mp3_dataset

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