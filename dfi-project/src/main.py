import argparse

from __builtin__ import xrange

from DFI import DFI


def parse_arg():
    """Parse commandline arguments
    :return: argument dict
    """
    parser = argparse.ArgumentParser('Deep Feature Interpolation')
    parser.add_argument('--data_dir', '-d', default='data', type=str,
                        help='Path to data directory containing the images')
    parser.add_argument('--model_path', '-m', default='model/vgg19.npy', type=str,
                        help='Path to the model file (*.npy)')
    parser.add_argument('--gpu', '-g', default=False, action='store_true', help='Enable gpu computing')
    parser.add_argument('--num_layers', '-n', default=3, type=int, help='Number of layers. One of {1,2,3}')
    args = vars(parser.parse_args())

    if args['num_layers'] not in xrange(1, 4):
        raise argparse.ArgumentTypeError("%s is an invalid int value. (1 <= n <= 3)" % args['num_layers'])

    return args


def main():
    """
    Main method
    :return: None
    """
    # Get args
    args = parse_arg()

    # Init DFI and run
    dfi = DFI(**args)
    dfi.run()


if __name__ == '__main__':
    main()
