"""--optim.name muon: parallel/muon.py's MuonWithAdamW. CPU: the update equals torch.optim.Muon (2-D hidden
weights, Moonlight's lr scale) + torch.optim.AdamW (the rest) exactly, and a state_dict round trip resumes exactly. GPU
(2 ranks, FSDP2 row shards incl. an uneven one): the sharded step equals the unsharded reference."""
import copy
import os
import types

import pytest
import torch
import torch.nn as nn

from pretrain.parallel.muon import MuonWithAdamW, muon_param_groups
from _helpers import free_port

CFG = types.SimpleNamespace(lr=2.4e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, no_wd_regex=(r"norm", r"embed_tokens", r"\.bias$"),
                            muon_exclude_regex=(r"embed_tokens", r"lm_head"), muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5,
                            muon_adjust_lr="match_rms_adamw")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, 32)
        self.fc1 = nn.Linear(32, 51, bias=False)       # 51 rows: an uneven 2-way row split
        self.fc2 = nn.Linear(51, 32, bias=True)
        self.norm = nn.LayerNorm(32)
        self.lm_head = nn.Linear(32, 64, bias=False)

    def forward(self, x):
        return self.lm_head(self.norm(self.fc2(torch.relu(self.fc1(self.embed_tokens(x))))))


def _build(model):
    groups, names = muon_param_groups(model, CFG)
    return MuonWithAdamW(groups, lr=CFG.lr, betas=CFG.betas, eps=CFG.eps, weight_decay=CFG.weight_decay), names


def _reference(model):
    """torch's own Muon + AdamW on the same split."""
    groups, _ = muon_param_groups(model, CFG)
    mu = [g for g in groups if g["use_muon"]][0]["params"]
    ad = [{"params": g["params"], "weight_decay": g["weight_decay"]} for g in groups if not g["use_muon"]]
    return (torch.optim.Muon(mu, lr=CFG.lr, weight_decay=CFG.weight_decay, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(ad, lr=CFG.lr, betas=CFG.betas, eps=CFG.eps, weight_decay=CFG.weight_decay))


def _loss(model, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 64, (4, 12), generator=g)
    return nn.functional.cross_entropy(model(x).flatten(0, 1), torch.randint(0, 64, (48,), generator=g))


def test_split():
    _, names = _build(Tiny())
    assert names["muon"] == ["fc1.weight", "fc2.weight"]
    assert set(names["decay"]) == {"lm_head.weight"} and "embed_tokens.weight" in names["no_decay"] and "fc2.bias" in names["no_decay"]


def test_matches_torch_muon_adamw_cpu():
    torch.manual_seed(0)
    a = Tiny(); b = copy.deepcopy(a)
    opt, _ = _build(a); ref = _reference(b)
    for step in range(4):
        for m, os_ in ((a, (opt,)), (b, ref)):
            for o in os_:
                o.zero_grad()
            _loss(m, step).backward()
            for o in os_:
                o.step()
        for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
            assert torch.equal(p, q), (step, n, (p - q).abs().max().item())


def test_state_dict_roundtrip_cpu():
    torch.manual_seed(1)
    a = Tiny(); opt, _ = _build(a)
    for step in range(2):
        opt.zero_grad(); _loss(a, step).backward(); opt.step()
    b = copy.deepcopy(a); opt_b, _ = _build(b)
    opt_b.load_state_dict(copy.deepcopy(opt.state_dict()))
    for m, o in ((a, opt), (b, opt_b)):
        o.zero_grad(); _loss(m, 7).backward(); o.step()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n


def _fsdp_worker(rank, world, port, out):
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    ref_model = Tiny().cuda()
    model = copy.deepcopy(ref_model)
    for mod in (model.fc1, model.fc2, model.lm_head):
        fully_shard(mod)
    fully_shard(model)
    opt, _ = _build(model)
    ref = _reference(ref_model)
    for step in range(3):
        opt.zero_grad(); _loss_cuda(model, step).backward(); opt.step()          # same batch on every rank: the mean grad = the grad
        for o in ref:
            o.zero_grad()
        _loss_cuda(ref_model, step).backward()
        for o in ref:
            o.step()
    diffs = {}
    for (n, p), (_, q) in zip(model.named_parameters(), ref_model.named_parameters()):
        full = p.full_tensor() if hasattr(p, "full_tensor") else p
        diffs[n] = (full - q).abs().max().item()
    if rank == 0:
        torch.save(diffs, out)
    dist.destroy_process_group()


def _loss_cuda(model, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 64, (4, 12), generator=g).cuda()
    y = torch.randint(0, 64, (48,), generator=g).cuda()
    return nn.functional.cross_entropy(model(x).flatten(0, 1), y)


@pytest.mark.gpu
def test_fsdp2_muon_matches_unsharded(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    import torch.multiprocessing as mp
    out = str(tmp_path / "diffs.pt")
    mp.spawn(_fsdp_worker, args=(2, free_port(), out), nprocs=2, join=True)
    diffs = torch.load(out)
    assert set(diffs) == {n for n, _ in Tiny().named_parameters()}
    bad = {n: d for n, d in diffs.items() if d > 1e-6}
    assert not bad, bad
