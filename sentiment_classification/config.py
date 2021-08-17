import argparse


def load_config():
    """CLI for program"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of epochs")

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help="Learning rate")

    parser.add_argument('--save_path', type=str, default="trained/")

    args = parser.parse_args()

    return args
