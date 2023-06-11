import os, random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data
from option import args
from model import fmen
import loss
from loss import *
import utility
import math
import pdb

from model import edsr, rcan, san, rlfn_s

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device('cpu' if args.cpu else 'cuda')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_teachers():
    print("Loading Teacher ====================================>")
    teachers = []

    if "EDSR" in args.teacher:
        args.n_feats = 256
        args.n_resblocks = 32
        args.res_scale = 0.1
        net = edsr.EDSR(args).to(device)
        if int(args.scale[0]) == 2:
            print("loading EDSRx2")
            net.load_state_dict(torch.load('./pretrained_models/pretrained-EDSR/EDSR_x2.pt'))
        elif int(args.scale[0]) == 3:
            print("loading EDSRx3")
            net.load_state_dict(torch.load('./pretrained_models/pretrained-EDSR/EDSR_x3.pt'))
        elif int(args.scale[0]) == 4:
            print("loading EDSRx4")
            net.load_state_dict(torch.load('./pretrained_models/pretrained-EDSR/EDSR_x4.pt'))
        if args.precision == 'half':
            net.half()
        teachers.append(net)
        print('=' * 40)
    if "RCAN" in args.teacher:
        args.n_resblocks = 20
        args.n_resgroups = 10
        net = rcan.RCAN(args).to(device)
        if int(args.scale[0]) == 2:
            print("loading RCANx2")
            net.load_state_dict(torch.load('./pretrained_models/models_ECCV2018RCAN/RCAN_BIX2.pt'))
        elif int(args.scale[0]) == 3:
            print("loading RCANx3")
            net.load_state_dict(torch.load('./pretrained_models/models_ECCV2018RCAN/.pt'))
        elif int(args.scale[0]) == 4:
            print("loading RCANx4")
            net.load_state_dict(torch.load('./pretrained_models/models_ECCV2018RCAN/RCAN_BIX4.pt'))
        elif int(args.scale[0]) == 8:
            print("loading RCANx8")
            net.load_state_dict(torch.load('./pretrained_models/models_ECCV2018RCAN/RCAN_BIX8.pt'))
        if args.precision == 'half':
            net.half()
        teachers.append(net)
        print('=' * 40)
   
    for teacher in teachers:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
  
    for teacher in teachers:
        print("+" * 10, "Teacher model size", "+" * 10)
        print(get_parameter_number(teacher))

    return teachers


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def print_model(model):
    print('===> model summary')
    try:
        print(get_parameter_number(model))
        from torchsummary import summary
        summary(model, (3, int(args.patch_size / args.scale[0]), int(args.patch_size / args.scale[0])), depth=1, verbose=1,
                col_names=["input_size", "output_size", "kernel_size", "num_params", "mult_adds"])
    except ImportError as error:
        # Output expected ImportErrors.
        print(error.__class__.__name__ + ": " + error.message)
        print('Need install torch-summary from "pip install torch-summary" ')
        print(model)
    except Exception:
        print('Need install torch-summary from "pip install torch-summary" not "torchsummary"')
        print(model)


def create_student_model():
    print("Preparing Student ===================================>")
    student_checkpoint = utility.checkpoint(args)

    student = rlfn_s.RLFN_S().to(device) 



    if args.precision == 'half':
        student.half()

    
    total_params = sum(p.numel() for p in student.parameters())
    print("+" * 10, "Student model size", "+" * 10)
    print(get_parameter_number(student))
    print(f'{total_params:,} total parameters in student model.')
    
    if args.resume:
        load_from = os.path.join(student_checkpoint.dir, 'model', 'model_latest.pt')
        student.load_state_dict(torch.load(load_from))
    if args.warmup:
        student.load_state_dict(torch.load(args.dir_warmup))

    return student_checkpoint, student


def prepare_criterion():
    criterion = Loss(args, student_ckp)
    if args.resume:
        criterion.load(student_ckp.dir)
    return criterion


def prepare_optimizer():
    optimizer = utility.make_optimizer(args, student)
    if args.resume:
        optimizer.load(student_ckp.dir, epoch=len(student_ckp.log))
    return optimizer


def teacher_selector(teachers):
    idx = np.random.randint(len(teachers))
    return teachers[idx], idx


def prepare(lr, hr):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(lr), _prepare(hr)]


def train(epoch):
    optimizer.schedule()
    criterion.step()
    student.train()
    criterion.start_log()
    CL = args.CL
    neg_num = 10

    lr = optimizer.get_lr()

    student_ckp.write_log(
        '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, lr)
    )

    timer_data, timer_model = utility.timer(), utility.timer()
    for batch, (lr, hr, _) in enumerate(train_loader):

        lr, hr = prepare(lr, hr)
        timer_data.hold()
        timer_model.tic()
        optimizer.zero_grad()
        student_sr = student(lr)
        # display which we are using now
        teacher, teacher_index = teacher_selector(teachers)
        teacher_sr = teacher(lr)

        dis_loss = criterion(student_sr, teacher_sr, hr)

        bic_sample = lr[torch.randperm(neg_num), :, :, :]
        bic_sample = upsampler(bic_sample)

        total_loss = dis_loss
        criterion.log[-1, 2] += total_loss.item()
        total_loss.backward()
        optimizer.step()

        timer_model.hold()

        if (batch) % args.print_every == 0:
            student_ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                (batch) * args.batch_size,
                len(train_loader.dataset),
                criterion.display_loss(batch),
                timer_model.release(),
                timer_data.release()))

        timer_data.tic()

    criterion.end_log(len(train_loader))


def test(epoch):
    student.eval()
    with torch.no_grad():
        if args.save_results:
            student_ckp.begin_background()

        student_ckp.write_log('\nEvaluation:')
        student_ckp.add_log(torch.zeros(1, len(args.scale)))

        timer_test = utility.timer()

        for idx_scale, scale in enumerate(args.scale):
            eval_acc = 0
            test_loader.dataset.set_scale(idx_scale)
            tqdm_test = tqdm(test_loader, ncols=80) 
            for idx_data, (lr, hr, filename) in enumerate(tqdm_test):
                lr, hr = prepare(lr, hr)
                sr = student(lr)
                sr = utility.quantize(sr, args.rgb_range)

                save_list = [sr]
                eval_acc += utility.calc_psnr(
                    sr, hr, scale, args.rgb_range,
                    benchmark=test_loader.dataset.benchmark
                )
                if args.save_gt:
                    save_list.extend([lr, hr])
                if args.save_results:
                    student_ckp.save_results(filename, save_list, scale)

            student_ckp.log[-1, idx_scale] = eval_acc / len(test_loader)
            best = student_ckp.log.max(0)

            student_ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    args.data_test,
                    scale,
                    student_ckp.log[-1, idx_scale],
                    best[0][idx_scale],
                    best[1][idx_scale] + 1
                )
            )
        student_ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))

        if args.save_results:
            student_ckp.end_background()

        save(is_best=(best[1][0] + 1 == epoch), epoch=epoch)

        student_ckp.write_log('Saving Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)


def save(is_best, epoch):
    save_root_path = student_ckp.dir

    # save model
    save_dirs = [os.path.join(save_root_path, 'model', 'model_latest.pt')]
    if is_best:
        save_dirs.append(os.path.join(save_root_path, 'model', 'model_best.pt'))
    if args.save_models:
        save_dirs.append(os.path.join(save_root_path, 'model', 'model_{}.pt'.format(epoch)))
    for s in save_dirs:
        torch.save(student.state_dict(), s)

    # save loss
    criterion.save(save_root_path)
    criterion.plot_loss(save_root_path, epoch)

    # save optimizer
    optimizer.save(save_root_path)

    # save psnr
    student_ckp.plot_psnr(epoch)
    torch.save(student_ckp.log, os.path.join(save_root_path, 'psnr_log.pt'))


def print_args():
    msg = ""
    msg += "Model settings\n"
    msg += "Teachers: %s\n" % args.teacher
    msg += "Student: %s\n" % args.model

    msg += "\n\n"

    return msg


if __name__ == "__main__":
    seed_everything(100)
    msg = print_args()

    print("Preparing Data ====================================>")
    loader = data.Data(args)
    train_loader = loader.loader_train
    test_loader = loader.loader_test

    teachers = load_teachers()
    student_ckp, student = create_student_model()

    criterion = prepare_criterion()
    optimizer = prepare_optimizer()
    upsampler = nn.Upsample(scale_factor=args.scale[0], mode='bicubic')
    
    vgg_weight = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
    d_func = "L1"
    t_detach = True
    contra_loss = ContrastLoss(vgg_weight, d_func, t_detach)

    student_ckp.write_log(msg)

    epoch = 1
    if args.resume == 1:
        # pass
        epoch = len(student_ckp.log) + 1

    print("Start Training ======================================>")
    while epoch < args.epochs + 1:
        train(epoch)
        test(epoch)
        epoch += 1


