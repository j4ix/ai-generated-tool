import argparse
from train import train

# define command-line arguments
parser = argparse.ArgumentParser(description='Train GAN on MP3 data')
parser.add_argument('--data-dir', type=str, required=True, help='Path to directory containing MP3 data')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train for')
parser.add_argument('--noise-dim', type=int, default=100, help='Dimension of input noise for generator')
parser.add_argument('--seed-mp3s', type=str, nargs='+', help='List of paths to seed MP3s for generating samples')
parser.add_argument('--output-dir', type=str, default='output', help='Path to directory for saving generated samples')

# parse command-line arguments
args = parser.parse_args()

# train GAN model
train(data_dir=args.data_dir, batch_size=args.batch_size, epochs=args.epochs, noise_dim=args.noise_dim,
      seed_mp3s=args.seed_mp3s, output_dir=args.output_dir)
