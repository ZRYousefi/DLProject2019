import re
from matplotlib import pyplot as plt
import numpy as np

def extract_loss(input_file):
	"""
	Extract loss values from log file
	Each line has Loss_d_metric: <loss_value>
	Then using regex we can extract them
	"""
	with open(input_file) as f:
		return [float(m.group(1)) for m in 
				re.finditer(r'Loss_d_metric:\s(\d+.\d+)+', f.read())]


def visualize_loss(input_file, sampling_rate=10, aspect_ratio=1, **kargs):
	"""
	Visualize the losses. Becase steps are too much, we are getting samples
	with sampling_rate
	"""
	plt.figure(figsize=(20, 7))
	losses = extract_loss(input_file)
	plt.tight_layout()
	if 'mini' in input_file.lower():
		title = 'Mini-imagenet'
	elif 'omni' in input_file.lower():
		title = 'Omniglot'
	elif 'food' in input_file.lower():
		title = 'Food-101'
	
	plt.suptitle(title, fontsize=20)
	plt.xlabel('Iteration', fontsize=18)
	plt.ylabel('Loss value', fontsize=18)
	return plt.plot(np.arange(1, len(losses), sampling_rate), losses[::sampling_rate], **kargs)


if __name__ == '__main__':
	f_name = 'checkpoints/food101_N20_S1_10h/run_food.log'
	f = visualize_loss(f_name, 10)
	# plt.show()
	plt.savefig('%s.pdf' % (f_name.split('/')[-1].split('.')[0]))