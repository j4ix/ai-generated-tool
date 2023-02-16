import tensorflow as tf
from data import load_data, preprocess_data
from model import WaveGANGenerator, WaveGANDiscriminator

# Define training parameters
BATCH_SIZE = 32
EPOCHS = 1000
SEED_MP3S = ["seed1.mp3", "seed2.mp3", "seed3.mp3"]

# Load and preprocess data
data = load_data("path/to/mp3/files")
train_data, valid_data = preprocess_data(data, BATCH_SIZE)

# Define model and optimizer
generator = WaveGANGenerator()
discriminator = WaveGANDiscriminator()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Define loss function
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch}/{EPOCHS}")
    for batch in train_data:
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake audio using generator
            fake_audio = generator(tf.random.normal([BATCH_SIZE, 100]))

            # Train discriminator on real audio
            real_audio = batch
            real_output = discriminator(real_audio)
            # Train discriminator on fake audio
            fake_output = discriminator(fake_audio)

            # Calculate discriminator loss
            disc_loss = loss_fn(tf.ones_like(real_output), real_output) + \
                        loss_fn(tf.zeros_like(fake_output), fake_output)

            # Calculate generator loss
            gen_loss = loss_fn(tf.ones_like(fake_output), fake_output)

            # Calculate gradients and apply to optimizer
            gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
            optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

# Print loss values and other metrics
print(f"Generator loss: {gen_loss:.4f}")
print(f"Discriminator loss: {disc_loss:.4f}")

# Generate new mp3 every 100 epochs
if epoch % 100 == 0:
    seed_audio = [load_audio_file(mp3_file) for mp3_file in SEED_MP3S]
    generated_audio = generator(tf.random.normal([1, 100]))
    save_audio_file(generated_audio, f"generated_audio_epoch{epoch}.mp3")
