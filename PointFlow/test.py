from copy import deepcopy
from datasets import get_datasets, synsetid_to_cate
from args import get_args
from pprint import pprint
from pointcloud_dataset import PointFlowDataset
from core.utils.core_utils import VArgs
import pickle
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from PointFlow.utils import visualize_point_clouds
from collections import defaultdict
from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn
import imageio


def get_test_loader(args, train_stat):
    args = deepcopy(args)
    # args.data_dirs = ['./datasets/0323_multicut_train/', './datasets/0323_multimerge_train/']
    args.load_from_buffer = True
    args.sample_size = args.tr_max_sample_points
    args.seed = 0
    args.split = 'val'
    args.train_std = train_stat['std']
    te_dataset = PointFlowDataset(args)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def evaluate_recon(model, args, loader):
    # TODO: make this memory efficient
    if 'all' in args.cates:
        cates = list(synsetid_to_cate.values())
    else:
        cates = args.cates
    all_results = {}
    cate_to_len = {}
    save_dir = os.path.dirname(args.resume_checkpoint)
    for cate in cates:
        args.cates = [cate]

        all_sample = []
        all_ref = []
        results = []
        for data in loader:
            idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
            te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
            tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
            B, N = te_pc.size(0), te_pc.size(1)
            out_pc = model.reconstruct(tr_pc, num_points=N)
            m, s = data['mean'].float(), data['std'].float()
            m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
            s = s.cuda() if args.gpu is None else s.cuda(args.gpu)

            results = []
            for idx in range(5):
                res = visualize_point_clouds(out_pc[idx], tr_pc[idx], idx, args.dataset_type,
                                             pert_order=loader.dataset.display_axis_order)
                results.append(res)
            out_pc = out_pc * s + m
            te_pc = te_pc * s + m

            all_sample.append(out_pc)
            all_ref.append(te_pc)
        res = np.concatenate(results, axis=1)
        imageio.imwrite(os.path.join(save_dir, 'images', 'reconstruct.png'), res.transpose((1, 2, 0)))
        sample_pcs = torch.cat(all_sample, dim=0)
        ref_pcs = torch.cat(all_ref, dim=0)
        cate_to_len[cate] = int(sample_pcs.size(0))
        print("Cate=%s Total Sample size:%s Ref size: %s"
              % (cate, sample_pcs.size(), ref_pcs.size()))

        # Save it
        np.save(os.path.join(save_dir, "%s_out_smp.npy" % cate),
                sample_pcs.cpu().detach().numpy())
        np.save(os.path.join(save_dir, "%s_out_ref.npy" % cate),
                ref_pcs.cpu().detach().numpy())

        results = EMD_CD(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
        results = {
            k: (v.cpu().detach().item() if not isinstance(v, float) else v)
            for k, v in results.items()}
        pprint(results)
        all_results[cate] = results

    # Save final results
    print("=" * 80)
    print("All category results:")
    print("=" * 80)
    pprint(all_results)
    save_path = os.path.join(save_dir, "percate_results.npy")
    np.save(save_path, all_results)

    # Compute weighted performance
    ttl_r, ttl_cnt = defaultdict(lambda: 0.), defaultdict(lambda: 0.)
    for catename, l in cate_to_len.items():
        for k, v in all_results[catename].items():
            ttl_r[k] += v * float(l)
            ttl_cnt[k] += float(l)
    ttl_res = {k: (float(ttl_r[k]) / float(ttl_cnt[k])) for k in ttl_r.keys()}
    print("=" * 80)
    print("Averaged results:")
    pprint(ttl_res)
    print("=" * 80)

    save_path = os.path.join(save_dir, "results.npy")
    np.save(save_path, all_results)


def evaluate_gen(model, args, loader):
    all_sample = []
    all_ref = []
    for data in loader:
        idx_b, te_pc = data['idx'], data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc = model.sample(B, N)

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

        sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))

    # Save the generative output
    save_dir = os.path.dirname(args.resume_checkpoint)
    np.save(os.path.join(save_dir, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy())
    np.save(os.path.join(save_dir, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())

    # Compute metrics
    results = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)

    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)


def main(args):
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint_path = args.resume_checkpoint
    checkpoint = torch.load(args.resume_checkpoint)

    # Update training args about integration time
    variant_path = os.path.join(os.path.dirname(args.resume_checkpoint), 'variant.json')
    import json
    with open(variant_path, 'r') as f:
        vv = json.load(f)
    args = VArgs(vv)
    args.resume_checkpoint = checkpoint_path

    if not vv['train_T']:
        args.train_T = False
        args.time_length = vv['time_length']
        # checkpoint['point_cnf.module.chain.1.sqrt_end_time'] = torch.ones(1) * np.sqrt(vv['time_length'])
        # checkpoint['latent_cnf.module.chain.1.sqrt_end_time'] = torch.ones(1) * np.sqrt(vv['time_length'])

    if 'model' in checkpoint:
        checkpoint = checkpoint['model']

    model = PointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    model.load_state_dict(checkpoint)
    model.eval()

    # Load training dataset statistics
    stat_path = os.path.join(os.path.dirname(checkpoint_path), 'train_stat.pkl')
    with open(stat_path, 'rb') as f:
        stat = pickle.load(f)
    print("Loading stats")

    loader = get_test_loader(args, stat)

    with torch.no_grad():
        # Evaluate reconstruction
        evaluate_recon(model, args, loader)
        # Evaluate generation
        evaluate_gen(model, args, loader)


if __name__ == '__main__':
    args = get_args()
    main(args)
