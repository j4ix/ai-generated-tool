import os
import tensorflow as tf
import tensorflow_io as tfio
from pydub import AudioSegment

def load_mp3_dataset(data_dir, batch_size=64, shuffle_buffer_size=10000):
    # get list of MP3 file paths
    filepaths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp3'):
                filepaths.append(os.path.join(root, file))

    # create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(filepaths)

    # shuffle file paths
    dataset = dataset.shuffle(len(filepaths))

    # decode MP3 files and pad to 128k samples
    @tf.function
    def decode_mp3(file_path):
        file_path = tf.strings.join([data_dir, '/', file_path])
        audio = AudioSegment.from_file(file_path.numpy(), format="mp3")
        audio = audio.set_channels(1)  # convert stereo to mono
        audio = audio.get_array_of_samples()
        audio = tf.constant(audio, dtype=tf.float32) / 32768.0
        audio = tf.expand_dims(audio, axis=-1)
        audio = tf.image.resize_with_pad(audio, target_height=128000, target_width=1)
        audio = tf.squeeze(audio, axis=-1)
        return audio

    def load_mp3_wrapper(file_path):
        return tf.py_function(decode_mp3, [file_path], tf.float32)

    dataset = dataset.map(load_mp3_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # batch and shuffle dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
