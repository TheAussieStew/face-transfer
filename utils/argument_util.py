
def get_args(parser):
    parser.add_argument('--batch_size', nargs="?", type=int, default=8, help='batch_size for experiment')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="face_transfer_autoencoder", help='Experiment name')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='continue from checkpoint of epoch')
    parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='num of gpus to train on')
    parser.add_argument('--dropout_rate_value', nargs="?", type=float, default=0.0, help='dropout_rate_value')
    args = parser.parse_args()

    batch_size = args.batch_size
    num_gpus = args.num_of_gpus
    continue_from_epoch = args.continue_from_epoch
    dropout_rate_value = args.dropout_rate_value
    experiment_name = args.experiment_name

    return batch_size, num_gpus, continue_from_epoch, experiment_name, dropout_rate_value