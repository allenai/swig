import os, sys
sys.path.append(os.path.abspath("."))
import argparse
import json
import torch
import torch.optim as optim
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import model
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

from global_utils.imsitu_eval import BboxEval
from global_utils.format_utils import cmd_to_title
from global_utils import format_utils
import sys
print('CUDA available: {}'.format(torch.cuda.is_available()))



def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	parser.add_argument('--train-file', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--classes-file', help='Path to file containing class list (see readme)')
	parser.add_argument('--val-file', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--role-file', help='Path to role file')
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)
	parser.add_argument('--title', type=str, default='')
	parser.add_argument("--resume-epoch", type=int, default=0)
	parser.add_argument("--detach-epoch", type=int, default=12)
	parser.add_argument("--gt-noun-epoch", type=int, default=5)
	parser.add_argument("--hidden-size", type=int, default=1024)
	parser.add_argument("--lr-decrease", type=int, default=10)
	parser.add_argument("--second-lr-decrease", type=int, default=20)
	parser.add_argument("--iteration", type=float, default=100.0)
	parser.add_argument("--lr", type=float, default=.0006)
	parser.add_argument("--batch-size", type=int, default=16)
	parser = parser.parse_args(args)

	writer, log_dir = init_log_dir(parser)

	print('correct version')

	print("loading dev")
	with open('./SWiG_jsons/dev.json') as f:
		dev_gt = json.load(f)
	print("loading imsitu_dpace")
	with open('./SWiG_jsons/imsitu_space.json') as f:
		all = json.load(f)
		verb_orders = all['verbs']
		noun_dict = all['nouns']

	dataloader_train, dataset_train, dataloader_val, dataset_val = init_data(parser, verb_orders)
	print("loading model")
	retinanet = model.resnet50(num_classes=dataset_train.num_classes(), num_nouns=dataset_train.num_nouns(), parser=parser, pretrained=True)
	retinanet = torch.nn.DataParallel(retinanet).cuda()
	optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)

	print('weights loaded')

	for epoch_num in range(parser.resume_epoch, parser.epochs):
		train(retinanet, optimizer, dataloader_train, parser, epoch_num, writer)
		torch.save({'state_dict': retinanet.module.state_dict(), 'optimizer': optimizer.state_dict()}, log_dir + '/checkpoints/retinanet_{}.pth'.format(epoch_num))
		print('Evaluating dataset')
		evaluate(retinanet, dataloader_val, parser, dataset_val, dataset_train, verb_orders, dev_gt, epoch_num, writer, noun_dict)


def train(retinanet, optimizer, dataloader_train, parser, epoch_num, writer):
	retinanet.train()
	retinanet.module.freeze_bn()

	i = 0
	avg_class_loss = 0.0
	avg_reg_loss = 0.0
	avg_bbox_loss = 0.0
	avg_noun_loss = 0.0

	retinanet.training = True
	deatch_resnet = parser.detach_epoch > epoch_num
	use_gt_nouns = parser.gt_noun_epoch > epoch_num

	if epoch_num == parser.lr_decrease:
		lr = parser.lr / 10
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr

	if epoch_num == parser.second_lr_decrease:
		lr = parser.lr / 100
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr

	if use_gt_nouns:
		print("Using gt nouns")
	else:
		print("Not using gt nouns")

	for iter_num, data in enumerate(dataloader_train):
		i += 1
		optimizer.zero_grad()
		image = data['img'].cuda().float()
		annotations = data['annot'].cuda().float()
		verbs = data['verb_idx'].cuda()
		widths = data['widths'].cuda()
		heights = data['heights'].cuda()

		class_loss, reg_loss, bbox_loss, all_noun_loss = retinanet(image, annotations, verbs, widths, heights, epoch_num, deatch_resnet, use_gt_nouns)

		avg_class_loss += class_loss.mean().item()
		avg_reg_loss += reg_loss.mean().item()
		avg_bbox_loss += bbox_loss.mean().item()
		avg_noun_loss +=  all_noun_loss.mean().item()

		if i % parser.iteration == 0:

			print(
				'Epoch: {} | Iteration: {} | Class loss: {:1.5f} | Reg loss: {:1.5f} | Noun loss: {:1.5f} | Box loss: {:1.5f}'.format(
					epoch_num, iter_num, float(avg_class_loss / parser.iteration), float(avg_reg_loss / parser.iteration),
					float(avg_noun_loss / parser.iteration), float(avg_bbox_loss / parser.iteration)))
			writer.add_scalar("train/classification_loss", avg_class_loss / parser.iteration,
							  epoch_num * len(dataloader_train) + i)
			writer.add_scalar("train/regression_loss", avg_reg_loss / parser.iteration,
							  epoch_num * (len(dataloader_train)) + i)
			writer.add_scalar("train/bbox_loss", avg_bbox_loss / parser.iteration,
							  epoch_num * (len(dataloader_train)) + i)
			writer.add_scalar("train/noun_loss", avg_noun_loss / parser.iteration,
							  epoch_num * (len(dataloader_train)) + i)


			avg_class_loss = 0.0
			avg_reg_loss = 0.0
			avg_bbox_loss = 0.0
			avg_noun_loss = 0.0

		loss = class_loss.mean() + reg_loss.mean() + bbox_loss.mean() + all_noun_loss.mean()

		if bool(loss == 0):
			continue
		loss.backward()
		torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1, norm_type="inf")
		optimizer.step()


def evaluate(retinanet, dataloader_val, parser, dataset_val, dataset_train, verb_orders, dev_gt, epoch_num, writer, noun_dict):
	evaluator = BboxEval()
	retinanet.training = False
	retinanet.eval()
	k = 0
	for iter_num, data in enumerate(dataloader_val):
		if k % 100 == 0:
			print(str(k) + " out of " + str(len(dataset_val) / parser.batch_size))
		k += 1
		x = data['img'].cuda().float()
		y = data['verb_idx'].cuda()
		widths = data['widths'].cuda()
		heights = data['heights'].cuda()
		annotations = data['annot'].cuda().float()
		shift_1 = data['shift_1']
		shift_0 = data['shift_0']

		with torch.no_grad():
			verb_guess, noun_predicts, bbox_predicts, bbox_exists = retinanet(x, annotations, y, widths, heights, epoch_num, use_gt_verb=True)
		for i in range(len(verb_guess)):
			image = data['img_name'][i].split('/')[-1]
			verb = dataset_train.idx_to_verb[verb_guess[i]]
			nouns = []
			bboxes = []
			for j in range(6):
				if dataset_train.idx_to_class[noun_predicts[j][i]] == 'blank':
					nouns.append('')
				else:
					nouns.append(dataset_train.idx_to_class[noun_predicts[j][i]])
				if bbox_exists[j][i] > 0:
					bbox_predicts[j][i][0] = max(bbox_predicts[j][i][0] - shift_1[i], 0)
					bbox_predicts[j][i][1] = max(bbox_predicts[j][i][1] - shift_0[i], 0)
					bbox_predicts[j][i][2] = max(bbox_predicts[j][i][2] - shift_1[i], 0)
					bbox_predicts[j][i][3] = max(bbox_predicts[j][i][3] - shift_0[i], 0)
					bboxes.append(bbox_predicts[j][i] / data['scale'][i])
				else:
					bboxes.append(None)
			verb_gt, nouns_gt, boxes_gt = get_ground_truth(image, dev_gt[image], verb_orders)
			evaluator.update(verb, nouns, bboxes, verb_gt, nouns_gt, boxes_gt, verb_orders, 1)

	print(evaluator.verb())
	print(evaluator.value())
	print(evaluator.value_all())
	print(evaluator.value_bbox())
	print(evaluator.value_all_bbox())

	writer.add_scalar("val/verb_acc", evaluator.verb(), epoch_num)
	writer.add_scalar("val/value", evaluator.value(), epoch_num)
	writer.add_scalar("val/value_all", evaluator.value_all(), epoch_num)
	writer.add_scalar("val/value_bbox", evaluator.value_bbox(), epoch_num)
	writer.add_scalar("val/value_all_bbox", evaluator.value_all_bbox(), epoch_num)



def init_data(parser, verb_orders):
	dataset_train = CSVDataset(train_file=parser.train_file, class_list=parser.classes_file, verb_info= verb_orders, is_training=True,
							   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(True)]))

	if parser.val_file is None:
		dataset_val = None
		print('No validation annotations provided.')
	else:
		dataset_val = CSVDataset(train_file=parser.val_file, class_list=parser.classes_file, verb_info= verb_orders, is_training=False,
								 transform=transforms.Compose([Normalizer(), Resizer(False)]))

	sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=True)
	dataloader_train = DataLoader(dataset_train, num_workers=64, collate_fn=collater, batch_sampler=sampler)

	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=True)
		dataloader_val = DataLoader(dataset_val, num_workers=64, collate_fn=collater, batch_sampler=sampler_val)
	return dataloader_train, dataset_train, dataloader_val, dataset_val


def init_log_dir(parser):
	print()
	x = cmd_to_title(sys.argv[1:], True)
	print(x)
	log_dir = "./runs/" + x

	writer = SummaryWriter(log_dir)

	with open(log_dir + '/config.csv', 'w') as f:
		for item in vars(parser):
			print(item + ',' + str(getattr(parser, item)))
			f.write(item + ',' + str(getattr(parser, item)) + '\n')

	if not os.path.isdir(log_dir + "/checkpoints"):
		os.makedirs(log_dir + "/checkpoints")
	if not os.path.isdir(log_dir + '/map_files'):
		os.makedirs(log_dir + '/map_files')
	if parser.train_file is None:
		raise ValueError('Must provide --train-file when training,')
	if parser.classes_file is None:
		raise ValueError('Must provide --classes-file when training')
	return writer, log_dir



def get_ground_truth(image, image_info, verb_orders):
	verb = image.split("_")[0]
	nouns = []
	bboxes = []
	for role in verb_orders[verb]["order"]:
		all_options = set()
		for i in range(3):
			all_options.add(image_info["frames"][i][role])
		nouns.append(all_options)
		if image_info["bb"][role][0] == -1:
			bboxes.append(None)
		else:
			b = [int(i) for i in image_info["bb"][role]]
			bboxes.append(b)
	return verb, nouns, bboxes

if __name__ == '__main__':
 main()