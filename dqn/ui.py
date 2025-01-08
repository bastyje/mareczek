import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ALE/DonkeyKong-v5')
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--render', action='store_true')
    return parser.parse_args()