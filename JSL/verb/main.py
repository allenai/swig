import os, sys
sys.path.append(os.path.abspath("."))
import torch
from imsituDatasetGood import imSituDatasetGood
from torch.utils.data import DataLoader
from verbModel import ImsituVerb
from tensorboardX import SummaryWriter
import argparse
import datetime
import sys
import pdb
import os
import json


parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0001)



def main():
    #torch.multiprocessing.set_sharing_strategy('file_system')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.title = cmd_to_title(sys.argv[1:], True) if not args.title else args.title
    log_dir = "./runs/" + args.title
    writer = SummaryWriter(log_dir)

    checkpoint_dir = log_dir + "/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    verbs = './global_utils/verb_indices.txt'
    verb_to_idx, idx_to_verb = get_mapping(verbs)
    kwargs = {"num_workers": args.workers} if torch.cuda.is_available() else {}

    print("Loading Train Data")
    json_file = './SWiG_jsons/train.json'
    train_dataset = imSituDatasetGood(verb_to_idx, json_file, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


    print("Loading Test Data")
    json_file = './SWiG_jsons/dev.json'
    test_dataset = imSituDatasetGood(verb_to_idx, json_file, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, **kwargs)

    model = ImsituVerb()

    print(sum([p.numel() for p in model.parameters()]))

    model = torch.nn.DataParallel(model).cuda()

    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    checkpoint_dir = './runs/' + args.title + '/checkpoints'


    for epoch in range(100):
        if (epoch%18) == 0 and epoch > 0 and epoch < 25:
            lr = lr/5.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        train(model, train_dataloader, device, optimizer, writer, epoch)
        eval(model, test_dataloader, device, writer, epoch, idx_to_verb)
        save_name = checkpoint_dir + "/check_" + str(epoch)+ ".pth.tar"
        save_checkpoint({"state_dict": model.module.state_dict(), 'optimizer': optimizer.state_dict()}, save_name)


def train(model, data_loader, device, optimizer, writer, epoch_num):
    model.train()
    i = 0
    print()
    print('training epoch: ' + str(epoch_num))
    avg_loss = 0.0
    total = 0.0
    total_correct = 0.0
    for sample in data_loader:
        i += 1
        gt_verb, image_names, roles = (sample["verb"].to(device), sample["image"].to(device), sample["roles"].to(device))
        loss, verb = model(epoch_num, image_names, gt_verb)
        total += len(verb)
        total_correct += sum(gt_verb.squeeze() == verb)
        loss = loss.mean()
        avg_loss += loss.item()
        if (i % 100) == 0:
            writer.add_scalar("train/noun_loss", avg_loss / 100,
                              epoch_num * (len(data_loader)) + i)
            print("loss: " + str(avg_loss / 100))
            avg_loss = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(total_correct / total)
    writer.add_scalar('train/accuracy', total_correct / total, epoch_num)


def eval(model, data_loader, device, writer, epoch, idx_to_verb):
    model.eval()
    total = 0.0
    total_correct = 0.0
    print()
    print("eval")
    with open('top_5_verbs.json', 'w') as f:
        results = {}
        with torch.no_grad():
            for sample in data_loader:
                words = sample["im_name"]
                gt_verb, image_names, roles = (sample["verb"].to(device), sample["image"].to(device), sample["roles"].to(device))
                verb, top_5_verb = model(epoch, image_names, gt_verb, is_train=False)
                for i in range(len(words)):
                    top_verbs = []
                    for j in range(5):
                        top_verbs.append(str(idx_to_verb[int(top_5_verb[i][j])]))
                    #print(words[i])
                    #print(top_verbs)
                    results[words[i]] = top_verbs
                total += len(verb)
                total_correct += sum(gt_verb.squeeze() == verb)
            print(total_correct/total)
            writer.add_scalar('accuracy/lr', total_correct/total, epoch)
        json.dump(results, f)


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





