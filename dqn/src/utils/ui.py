import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ALE/DonkeyKong-v5')
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--obs-type', type=str, default='grayscale')
    parser.add_argument('--continue-from', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()