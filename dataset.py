import os
import tensorflow as tf
import pydub
import tempfile

def save_mp3(audio, file_path):
    audio = tf.cast(audio * 32768.0, tf.int16)
    audio = audio.numpy().flatten()
    audio = pydub.AudioSegment(audio.tobytes(), frame_rate=44100, sample_width=2, channels=1)
    audio.export(file_path, format='mp3')

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
    def decode_mp3(file_path):
        audio_binary = tf.io.read_file(file_path)

        # write audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(audio_binary.numpy())
            audio_file = f.name
 
        audio = pydub.AudioSegment.from_mp3(audio_file)
        audio = audio.set_channels(2)  # set to stereo
        audio = audio.get_array_of_samples()
        audio = tf.convert_to_tensor(audio, dtype=tf.float32) / 32768.0  # scale to [-1, 1]
        audio = tf.reshape(audio, (-1, 2))  # reshape to stereo
        audio = tf.expand_dims(audio, axis=-1)
        audio = tf.image.resize_with_pad(audio, target_height=128000, target_width=1)
        save_mp3(audio, file_path + ' test')
        return audio


    dataset = dataset.map(decode_mp3, num_parallel_calls=tf.data.AUTOTUNE)

    # batch and shuffle dataset
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
