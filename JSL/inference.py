import os, sys
sys.path.append(os.path.abspath("."))
from verb.imsituDatasetGood import imSituDatasetGood
from verb.verbModel import ImsituVerb
import datetime
import h5py
import argparse
import json
import torch
from torchvision import datasets, models, transforms
import JSL.gsr.model as jsl_model
from JSL.gsr.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
import pdb


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--detach_epoch", type=int, default=12)
    parser.add_argument("--gt_noun_epoch", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verb-path", type=str, default=None)
    parser.add_argument("--jsl-path", type=str, default=None)
    parser.add_argument("--image-file", type=str, default='')
    parser.add_argument("--store-features", action="store_true", default=False)

    args = parser.parse_args()

    if args.verb_path == None:
        print('please input a path to the verb model weights')
        return
    if args.jsl_path == None:
        print('please input a path to the jsl model weights')
        return
    if args.image_file == None:
        print('please input a path to the image file')
        return

    if args.store_features:
        if not os.path.exists('local_features'):
            os.makedirs('local_features')

    kwargs = {"num_workers": args.workers} if torch.cuda.is_available() else {}
    verbs = './global_utils/verb_indices.txt'
    verb_to_idx, idx_to_verb = get_mapping(verbs)

    print("initializing verb model")

    test_dataset = imSituDatasetGood(verb_to_idx, args.image_file, inference=True, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, **kwargs)
    model = ImsituVerb()
    model = torch.nn.DataParallel(model).cuda()
    x = torch.load(args.verb_path)
    model.module.load_state_dict(x['state_dict'])
    results = eval(model, test_dataloader, idx_to_verb, args)

    with open('./SWiG_jsons/imsitu_space.json') as f:
        all = json.load(f)
        verb_orders = all['verbs']
        noun_dict = all['nouns']

    print("initializing gsr model")

    dataset_val = CSVDataset(train_file=args.image_file, class_list='./global_utils/train_classes.csv', inference=True, inference_verbs=results,
                             verb_info=verb_orders, is_training=False, transform=transforms.Compose([Normalizer(), Resizer(False)]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=args.batch_size, drop_last=True)
    dataloader_val = DataLoader(dataset_val, num_workers=64, collate_fn=collater, batch_sampler=sampler_val)

    print("loading weights")
    retinanet = jsl_model.resnet50(num_classes=dataset_val.num_classes(), num_nouns=dataset_val.num_nouns(), parser=args, pretrained=True)
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    x = torch.load(args.jsl_path)
    retinanet.module.load_state_dict(x['state_dict'], strict = False)
    print('weights loaded')

    evaluate(retinanet, dataloader_val, args, dataset_val, dataset_val, verb_orders, noun_dict, idx_to_verb, args.store_features)


def evaluate(retinanet, dataloader_val, parser, dataset_val, dataset_train, verb_orders, noun_dict, idx_to_verb, store_features):
    retinanet.training = False
    retinanet.eval()
    k = 0
    print("predicting situation for each image...")
    results = {}
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
            if store_features:
                verb_guess, noun_predicts, bbox_predicts, bbox_exists, local_features = retinanet(x, annotations, y, widths, heights, 1, use_gt_verb=True, return_local_features=True)
            else:
                verb_guess, noun_predicts, bbox_predicts, bbox_exists = retinanet(x, annotations, y, widths, heights, 1, use_gt_verb=True)

        for i in range(len(verb_guess)):
            image = data['img_name'][i].split('/')[-1]
            verb = dataset_train.idx_to_verb[verb_guess[i]]
            nouns = []
            bboxes = []

            if store_features:
                just_image = image.split('.jpg')[0]
                features = h5py.File('local_features/{}.hdf5'.format(just_image), 'w')

            for j in range(len(verb_orders[verb]['order'])):

                if store_features:
                    noun_local_features = local_features[i, j].cpu()
                    features.create_dataset(str(j), data=noun_local_features)

                if dataset_train.idx_to_class[noun_predicts[j][i]] == 'blank':
                    nouns.append('')
                else:
                    nouns.append(dataset_train.idx_to_class[noun_predicts[j][i]])
                if bbox_exists[j][i] > 0:
                    bbox_predicts[j][i][0] = max(bbox_predicts[j][i][0] - shift_1[i], 0)
                    bbox_predicts[j][i][1] = max(bbox_predicts[j][i][1] - shift_0[i], 0)
                    bbox_predicts[j][i][2] = max(bbox_predicts[j][i][2] - shift_1[i], 0)
                    bbox_predicts[j][i][3] = max(bbox_predicts[j][i][3] - shift_0[i], 0)
                    bbb = bbox_predicts[j][i] / data['scale'][i]
                    bbb = [int(b) for b in bbb]
                    bboxes.append(bbb)
                else:
                    bboxes.append(None)
            if store_features:
                features.close()
            results[image] = {'verb': idx_to_verb[y[i]], 'nouns': nouns, 'boxes': bboxes}
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("complete. Results written to results.json")




def eval(model, data_loader, idx_to_verb, parser):
    model.eval()
    print()
    print("predicting verbs for each image...")
    results = {}
    k = 0
    with torch.no_grad():
        for sample in data_loader:
            if k % 100 == 0:
                print(str(k) + " out of " + str(len(data_loader) / parser.batch_size))
            words = sample["im_name"]
            image_names = (sample["image"].cuda())
            verb, top_5_verb = model(1, image_names, False, is_train=False)
            for i in range(len(words)):
                results[words[i]] = int(verb[i])
            k += 1
    print("verbs complete")
    return results


def get_mapping(word_file):
    dict = {}
    word_list = []
    with open(word_file) as f:
        k = 0
        for line in f:
            word = line.split('\n')[0]
            dict[word] = k
            word_list.append(word)
            k += 1
    return dict, word_list


def cmd_to_dict(cmd, list_format=False):
    """ Gets a dictionary of relevant args from a command. """
    if list_format:
        modified_cmd = cmd
        cmd = " ".join(cmd)
    else:
        modified_cmd = cmd.split()[2:]

    if "&" in modified_cmd:
        modified_cmd.remove("&")

    argmap = {}
    key = None

    for c in modified_cmd:
        if c[0] == "-":
            key = c[2:]
            argmap[key] = "true"
        else:
            argmap[key] = c

    argmap["cmd"] = cmd

    if "title" not in argmap:
        argmap["title"] = dict_to_title(argmap)

    return argmap


def dict_to_title(argmap):
    """ Converts a map of the relevant args to a title. """

    # python 3, this will be sorted and deterministic.
    print(argmap)
    return "_".join([k + "=" + v for k, v in argmap.items() if k != "cmd"])


def cmd_to_title(cmd, list_format=False):
    """ Gets a title from a command. """
    argmap = cmd_to_dict(cmd, list_format)
    return argmap["title"] + "_" + formatted_date()


def formatted_date():
    date = datetime.datetime.now()
    y = str(date).split(" ")
    return y[0] + "_" + y[1].split(".")[0]

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)
    print("Saved")


if __name__ == "__main__":
    main()





