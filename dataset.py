import tensorflow as tf

def load_and_preprocess_mp3(file_path):
    # TODO: load MP3 file and preprocess for GAN training
    pass

def load_mp3_dataset(data_dir):
    # list all MP3 files in data directory
    file_paths = tf.data.Dataset.list_files(data_dir + '/*.mp3')

    # load and preprocess MP3 files in parallel using all available CPU threads
    dataset = file_paths.map(load_and_preprocess_mp3, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
