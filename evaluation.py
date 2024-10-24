import data_loader
import datetime
import time
import torch
import models
import utils
from torch.autograd import Variable
from sys import path
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if __name__ == "__main__":
    print("Using device: {}".format(device))

    # Init submission Attacker
    from my_adv_attack import MyAdvAttack   # Attacker from submission

    # Start evaluation
    eps = 8/255
    val_loader = data_loader.get_loader()
    begin = time.time()         # <== Mark starting time
    results = {}
    n_samples = 10000           # We only evaluate 1000 selected samples on the server in phase 1, you may test with full test set
    limit = 100 * n_samples

    # Init models
    model_list = []

    # ResNet-18 with SAT
    model_0 = models.wrn.RobustWideResNet(num_classes=10, channel_configs=[64, 64, 128, 256, 512],
                                          depth_configs=[2, 2, 2, 2], stride_config=[1, 2, 2, 2], limit=limit)
    model_0.load_state_dict(torch.load('/Path/To/Your/Folder/RN-18-SAT.pth')['model_state_dict'], strict=False)
    model_list.append(model_0)

    for m_i,  model in enumerate(model_list):
        start = time.time()         # <== Mark starting time on a model
        adv_acc_meter = utils.AverageMeter()
        model = model.to(device)
        attacker = MyAdvAttack(model, eps)
        execution_success = True

        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Start evaluation for _model
        for images, labels in val_loader:
            # Submission generates adv example
            x_adv = attacker.perturb(images.detach().clone().to(device), labels.detach().clone().to(device)).cpu()

            assert x_adv.shape[0] == images.shape[0]

            # Ensure the x_adv within the eps norm
            eta = x_adv.data - images.data
            eta = torch.nan_to_num(eta, nan=0.0)
            eta = torch.clamp(eta, -eps, eps)
            x_adv = images + eta
            x_adv = Variable(torch.clamp(x_adv, 0, 1.0), requires_grad=False)

            # Evaluate Adv example
            model = model.eval()
            for param in model.parameters():
                param.requires_grad = False
            logits = model(x_adv.to(device)).cpu()
            acc = utils.accuracy(logits, labels)[0]
            adv_acc_meter.update(acc.item(), images.shape[0])

        end = time.time()
        time_spent = end - start
        payload = "Model {:d} Adversarial Error Rate: {:.6f} Runtime: {:5.2f} sec".format(
            m_i, 1 - adv_acc_meter.avg, time_spent)
        print(payload)

    exit(0)
