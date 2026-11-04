"""Server-only, 3-process CPU/Gloo numerical tests; no data or pretrained weights."""
import copy
from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch import nn

from event_state.training.engine import EventStateTrainer
from event_state.training.distributed import NullLogger
from event_state.training.optim import WarmupCosineScheduler
from event_state.training.data import StatefulStreamBatchSampler, MappedStreamSampler


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.event_encoder = nn.Linear(3, 4)
        self.temporal_model = nn.Linear(4, 4)
        self.z_projector = nn.Linear(4, 4)
        self.h_projector = nn.Linear(4, 4)

    def forward_sequence(self, events, state=None):
        z = self.event_encoder(events.mean((-1,-2))).unsqueeze(2)
        previous = torch.zeros_like(z[:,0]) if state is None else state[0]
        history = []
        for t in range(z.shape[1]):
            previous = torch.tanh(self.temporal_model(z[:,t]) + previous)
            history.append(previous)
        return dict(z=z, h=torch.stack(history, dim=1), state=previous.unsqueeze(0))

    def project_z(self, z): return self.z_projector(z)
    def project_h(self, h): return self.h_projector(h)


def trainer(model, cfg):
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    for group in opt.param_groups:
        group['initial_lr'] = .01
    return EventStateTrainer(model=model, teacher=None, train_loader=None, validation_loader=None,
        optimizer=opt, scheduler=WarmupCosineScheduler(opt, warmup_steps=0, total_steps=10, min_lr_ratio=.1),
        h_distillation_loss=nn.MSELoss(), z_distillation_loss=nn.MSELoss(), config=cfg,
        device=torch.device('cpu'), logger=NullLogger(), checkpoint_config=cfg)


class Repeat:
    epoch = 0
    def __init__(self, batch): self.batch = batch
    def __next__(self): return self.batch


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    # References are constructed outside the distributed group.
    refs, configs, models = [], [], []
    for h, z in [(False, 'direct_dino'), (True, 'none'), (True, 'direct_dino')]:
        torch.manual_seed(7)
        model = Tiny()
        cfg = dict(training=dict(precision='fp32', validation_enabled=False, gradient_accumulation=2),
                   teacher=dict(cache_features=True), model=dict(temporal=dict(detach_state_every=None)),
                   loss=dict(h_distill=dict(enabled=h), z_objective=dict(type=z, weight=1.)),
                   output=dict(directory=output))
        refs.append(trainer(copy.deepcopy(model), cfg))
        models.append(model)
        configs.append(cfg)
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous,
                                        rank=rank, world_size=3, timeout=timedelta(seconds=60))
    try:
        for condition, (model, cfg, reference) in enumerate(zip(models, configs, refs)):
            run = trainer(model, cfg)
            batches = []
            for r in range(3):
                torch.manual_seed(100+r)
                batches.append(dict(events=torch.randn(r+1,2,3,2,2), teacher_features=torch.randn(r+1,2,1,4),
                                    sequence_name=[f'r{r}_s{i}' for i in range(r+1)],
                                    is_sequence_start=torch.zeros(r+1, dtype=torch.bool)))
            full = {k:(sum([b[k] for b in batches], []) if k == 'sequence_name' else
                       torch.cat([b[k] for b in batches])) for k in batches[0]}
            # Both Hybrid branches, repeated updates, unequal rank batches and carried states.
            for _ in range(2):
                run.train_optimizer_step(Repeat(batches[rank]), Repeat(batches[rank]))
                reference.train_optimizer_step(Repeat(full), Repeat(full))
                for actual, expected in zip(model.parameters(), reference.model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            run._stream_state_bank = {f'sequence_{rank}': torch.ones(1,1,4)*rank}
            checkpoint = Path(output) / f'{condition}.pt'
            run._save(checkpoint, run._training_data_state(None, None))
            run.resume(checkpoint)
            assert set(run._restored_data_state['stream_state_bank']) == {f'sequence_{rank}'}
            if rank == 0:
                saved = torch.load(checkpoint, weights_only=False)
                assert saved['data_state']['world_size'] == 3
    finally:
        torch.distributed.destroy_process_group()


def test_three_rank_gradient_and_checkpoint(tmp_path):
    mp.spawn(worker, args=(str(tmp_path/'rendezvous'), str(tmp_path)), nprocs=3, join=True)


def test_mapped_stream_preserves_global_indices_and_epoch():
    records = [('a',1,2),('a',3,2),('b',1,2),('b',3,2)]
    mapped = MappedStreamSampler(StatefulStreamBatchSampler(records[:2], batch_size=1,
                                 seed=0, shuffle_sequences=False), [2,3])
    mapped.set_epoch(4)
    assert list(mapped) == [[(2,4)],[(3,4)]]
