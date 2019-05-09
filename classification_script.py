from classification import Classification
import sys
import argparse

parser = argparse.ArgumentParser(description='Script to train some DL models. When needed to run in detached state use "nohup python classification_script.py [args] &".')
parser.add_argument('--model', '-m', default='alexnet', help='Name of the model to be trained. Possible "alexnet" or "densenet"')
parser.add_argument('--batch-size', '-b', default=32, help='batch size')
parser.add_argument('--nbr-epochs', '-n', default=1, help='Maximum number of epochs to train model')
parser.add_argument('--save', '-s', default='parameters/default/', help='Location where the model parameter and metrics are saved')
parser.add_argument('--data-path', '-p', default='data/covers/', help='Location where the data can be loaded from')
parser.add_argument('--meta-path', default='data/meta/', help='Location where the label meta data can be loaded from')
args = parser.parse_args()

network_name = args.model
n_epochs = args.nbr_epochs
data_path = args.data_path
meta_path = args.meta_path
batch_size = args.batch_size
save_dir = args.save

classifier = Classification(data_path = data_path, meta_path = meta_path, batch_size = batch_size, save_dir=save_dir)

classifier.train(network_name=network_name, n_epochs = n_epochs, do_print = True)