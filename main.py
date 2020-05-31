import argparse
from train import train



def main(args):
	if args.train == "./train.py":
		train(args.batch_size, args.spatial_epochs, args.temporal_epochs, args.train_id, args.dB, args.spatial_size, args.flag, args.objective_flag, args.tensorboard)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', type=str, default='./train.py', help='Using which script to train.')
	parser.add_argument('--batch_size', type=int, default=16, help='Training Batch Size')
	parser.add_argument('--spatial_epochs', type=int, default=60, help='Epochs to train for Spatial Encoder')
	parser.add_argument('--temporal_epochs', type= int, default=20, help='Epochs to train for Temporal Encoder')
	parser.add_argument('--train_id', type=str, default="0", help='To name the weights of model')
	parser.add_argument('--dB', nargs="+", type=str, default='CASME2_TIM', help='Specify Database')
	parser.add_argument('--spatial_size', type=int, default=224, help='Size of image')
	parser.add_argument('--flag', type=str, default='st', help='Flags to control type of training')
	parser.add_argument('--objective_flag', type=int, default=1, help='Flags to use either objective class or emotion class')
	parser.add_argument('--tensorboard', type=bool, default=False, help='tensorboard display')


	args = parser.parse_args()
	print(args)

	main(args)