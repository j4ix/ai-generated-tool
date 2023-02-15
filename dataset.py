import tensorflow as tf

def load_mp3_dataset(data_dir, batch_size):
    file_pattern = f'{data_dir}/*.mp3'

    def decode_mp3(mp3):
        audio = tf.audio.decode_mp3(mp3)
        audio = tf.audio.encode_wav(tf.expand_dims(audio.audio, axis=-1), sample_rate=audio.sample_rate)
        audio = tf.audio.decode_wav(audio)
        audio = tf.image.resize(audio.audio, (128, 128))
        audio = tf.cast(audio, tf.float32) / 255.0
        return audio

    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.map(lambda x: tf.io.read_file(x))
    dataset = dataset.map(decode_mp3)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
