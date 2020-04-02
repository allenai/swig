import datetime
import numpy as np

def assign_learning_rate(optimizer, new_lr):
    print("assigned lr: " + str(new_lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length

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

    if "full_title" not in argmap:
        argmap["full_title"] = dict_to_title(argmap)

    return argmap


def dict_to_title(argmap):
    """ Converts a map of the relevant args to a title. """

    # python 3, this will be sorted and deterministic.
    print(argmap)
    exclude_list = ["train-file", "classes-file", "val-file", "cmd", "role-file"]
    return "_".join([k + "=" + v for k, v in argmap.items() if k not in exclude_list])


def cmd_to_title(cmd, list_format=False):
    """ Gets a title from a command. """
    argmap = cmd_to_dict(cmd, list_format)
    return argmap["full_title"] + "_" + formatted_date()


def formatted_date():
    date = datetime.datetime.now()
    y = str(date).split(" ")
    return y[0] + "_" + y[1].split(".")[0]