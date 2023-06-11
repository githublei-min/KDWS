import os
from tqdm import tqdm
# import numpy as np
import data
from option import args
from model import *
import utility
# import ipdb
# import gc
from model import rlfn_s, fmen


def prepare(lr, hr):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(lr), _prepare(hr)]

def test():
    model.eval()
    with torch.no_grad():
        for idx_scale, scale in enumerate(args.scale):
            eval_acc = 0
            test_loader.dataset.set_scale(idx_scale)
            tqdm_test = tqdm(test_loader, ncols=80)
            # psnr_list = []
            psnr = 0.
            ssim = 0.
            cc = 0.
            import time
            start = time.clock()
            for idx_data, (lr, hr, filename) in enumerate(tqdm_test):
                lr, hr = prepare(lr, hr)
                hr = hr.cpu()
                # print(lr.shape)
                #if lr.shape[3]>300:
                #    continue
                sr = model(lr).cpu()
                # ipdb.set_trace()
                sr = utility.quantize(sr, args.rgb_range)
                #save_list = [sr]
                psnr += utility.calc_psnr(sr, hr, scale, args.rgb_range, benchmark=test_loader.dataset.benchmark)
                ssim += utility.calc_ssim(sr, hr, scale, args.rgb_range, benchmark=test_loader.dataset.benchmark)
                cc += 1
                # print(psnr,ssim)
            elapsed = (time.clock() - start)
            print('average running time on ', args.data_test, ' dataset: ', elapsed)
            # print(psnr, ssim, cc)
            psnr /= cc #len(test_loader)
            ssim /= cc #len(test_loader)
            # psnr_list.append(psnr)
            # best_psnr = max(psnr_list)
            print('[{} x{}]\tPSNR: {:.4f}'.format(
                    args.data_test,
                    scale,
                    psnr
                 ))

            print('[{} x{}]\tSSIM: {:.4f}'.format(
                    args.data_test,
                    scale,
                    ssim
                 ))

    
if __name__ == '__main__':
    loader = data.Data(args)
    test_loader = loader.loader_test
    test_loader.dataset.set_scale(args.scale)
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
    device = torch.device('cpu' if args.cpu else 'cuda')

    sub_model = fmen.TRAIN_FMEN(args).to(device)
    # sub_model = rlfn_s.RLFN_S().to(device)
    sub_model.load_state_dict(torch.load(args.ckp_path))
    model = Model(sub_model, args.scale)
    test()

