import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

from train_age_groups_efficient import evaluate_joint_behavior, save_partial_best_snapshot, validate_cached_stage2_inputs
from vgg_accumulator_rnn import AccumulatorRNNDecision


def train_with_scale(scale, time_steps, train_cached, human_stats, epochs, device, hidden_dim, noise_std, threshold, choice_temperature, log_prefix=''):
    model = AccumulatorRNNDecision(n_classes=4, hidden_dim=hidden_dim, dt=10, time_steps=time_steps, threshold=threshold, noise_std=noise_std)
    model.input_scale.data.fill_(scale)
    model = model.to(device)
    logits_tensor = torch.tensor(train_cached['logits'], dtype=torch.float32)
    rts_tensor = torch.tensor(train_cached['rts_normalized'], dtype=torch.float32)
    target_tensor = torch.tensor(train_cached['target_labels'], dtype=torch.long)
    response_tensor = torch.tensor(train_cached['response_labels'], dtype=torch.long)
    congruency_tensor = torch.tensor(train_cached['congruency'], dtype=torch.long)
    dataloader = DataLoader(TensorDataset(logits_tensor, rts_tensor, target_tensor, response_tensor, congruency_tensor), batch_size=256, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-4)
    mse = torch.nn.MSELoss()
    best = None
    for epoch in range(epochs):
        model.train()
        for batch_logits, batch_rt, _batch_target, batch_response, batch_cong in dataloader:
            batch_logits = batch_logits.to(device)
            batch_rt = batch_rt.to(device)
            batch_response = batch_response.to(device)
            batch_cong = batch_cong.to(device)
            optimizer.zero_grad()
            decision_times, evidence_traj, threshold_t = model.rollout(batch_logits)
            threshold_t = torch.as_tensor(threshold_t, device=device, dtype=torch.float32)
            rt_loss = mse(decision_times.min(dim=1)[0], batch_rt)
            class_strength = (evidence_traj - threshold_t).amax(dim=1)
            choice_loss = F.cross_entropy(class_strength / choice_temperature, batch_response)
            if (batch_cong == 0).any() and (batch_cong == 1).any():
                mean_cong = decision_times.min(dim=1)[0][batch_cong == 0].mean()
                mean_incong = decision_times.min(dim=1)[0][batch_cong == 1].mean()
                cong_loss = F.relu(0.01 - (mean_incong - mean_cong))
            else:
                cong_loss = torch.tensor(0.0, device=device)
            loss = rt_loss + 3.0 * choice_loss + 0.3 * cong_loss
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_dt, evidence_traj, threshold_t = model.rollout(logits_tensor.to(device))
                threshold_t = torch.as_tensor(threshold_t, device=device, dtype=torch.float32)
                pred_rt = pred_dt.min(dim=1)[0].cpu().numpy()
                pred_choice = pred_dt.argmin(dim=1).cpu().numpy()
            results = evaluate_joint_behavior(pred_rt=pred_rt, pred_choice=pred_choice, true_rt=train_cached['rts'], target_labels=train_cached['target_labels'], response_labels=train_cached['response_labels'], congruency=train_cached['congruency'], human_stats=human_stats)
            print(f"{log_prefix}Eval epoch {epoch+1:02d}: score={results['total_score']:.4f} rt_score={results['rt_score']:.4f} acc={results['model_accuracy']:.4f}/{results['human_accuracy']:.4f} resp_agree={results['response_agreement']:.4f} cong_gap={results['model_congruency_rt_gap']:.4f}/{results['human_congruency_rt_gap']:.4f}")
            if best is None or results['behavior_optimal_score'] > best['results']['behavior_optimal_score']:
                best = {'epoch': epoch + 1, 'results': results, 'params': {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, 'pred_rt': pred_rt.copy(), 'pred_choice': pred_choice.copy()}
    if best is None:
        raise RuntimeError('No best checkpoint found')
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data_age_groups_matched')
    parser.add_argument('--output_root', default='checkpoints_age_groups_accumrnn')
    parser.add_argument('--age_group', default='20-29')
    parser.add_argument('--train_logits_path')
    parser.add_argument('--test_logits_path')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--scale_values', type=str, default='0.1,0.3,0.5')
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--time_steps', type=int, default=120)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--noise_std', type=float, default=0.02)
    parser.add_argument('--choice_temperature', type=float, default=0.10)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_root, args.age_group, 'stage2')
    data_dir = os.path.join(args.data_root, args.age_group)
    train_logits_path = args.train_logits_path or os.path.join(output_dir, 'train_logits.npz')
    test_logits_path = args.test_logits_path or os.path.join(output_dir, 'test_logits.npz')
    train_cached, test_cached = validate_cached_stage2_inputs(args.age_group, data_dir, train_logits_path, test_logits_path)
    human_stats = json.load(open(os.path.join(data_dir, 'rt_stats.json')))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    scales = np.array([float(x.strip()) for x in args.scale_values.split(',') if x.strip()], dtype=np.float32)
    best_overall = None
    for idx, scale in enumerate(scales, start=1):
        best = train_with_scale(scale, args.time_steps, train_cached, human_stats, args.epochs, device, args.hidden_dim, args.noise_std, args.threshold, args.choice_temperature, log_prefix=f'[accumrnn scale {idx}/{len(scales)}] ')
        if best_overall is None or best['results']['behavior_optimal_score'] > best_overall['results']['behavior_optimal_score']:
            best_overall = {'scale': scale, **best}
            save_partial_best_snapshot(output_dir, args.age_group, float(scale), best['epoch'], float(best['results']['total_score']), best['results'], best['params'], best['pred_rt'], best['pred_choice'], test_cached['target_labels'], test_cached['response_labels'], test_cached['congruency'], args.time_steps, args.data_root, args.output_root, 'response_labels')
    if best_overall is None:
        raise RuntimeError('No overall best checkpoint found')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump({'scale': float(best_overall['scale']), 'epoch': int(best_overall['epoch']), 'time_steps': int(args.time_steps), 'results': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in best_overall['results'].items()}}, f, indent=2)
    np.savez(os.path.join(output_dir, 'best_model_params.npz'), **best_overall['params'])
    print(f"Best scale for {args.age_group}: {best_overall['scale']:.3f}")
    print(f"Score: {best_overall['results']['total_score']:.4f}")
    print(f"Pred Mean RT: {best_overall['results']['pred_mean']:.3f}s")


if __name__ == '__main__':
    main()
