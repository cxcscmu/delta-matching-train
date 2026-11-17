#!/usr/bin/env python
"""Pretraining entry point (DDP / FSDP2).

Launch:
    torchrun --standalone --nproc_per_node=<N> -m pretrain.train --config cfg.yaml [--key=val ...]
    # or single process (1 GPU / CPU):
    python -m pretrain.train --config cfg.yaml

Flow: dist_init -> seed -> data -> model -> wrap (ddp|fsdp2) -> Trainer -> train.
Model init uses the SAME seed on every rank so the replicas/shards are identical at
step 0.

W&B is initialized only once the Trainer is built, i.e. after the resume was decided, checked (run identity, kernel
build) and loaded: a refused resume or a failed load never touches the run's W&B id. The end, in this order
(_close_and_exit): the process group is closed, then the master closes its W&B run, then the process exits. A slow W&B
upload therefore holds only the master, never the other ranks in the teardown's barrier.

Exit status: 0 = training complete (final checkpoint + export); 75 = stopped early (a SIGTERM / SIGUSR1, or the time
budget DM_STOP_BEFORE): the newest complete checkpoint holds the state, relaunch to resume (also when the stop came
before the first step, and whatever the process-group teardown or W&B do); 2 = the resume was refused for a reason no
retry can fix (the run identity, <ckpt_dir>/run_identity.json, or the checkpoint's kernel build differs;
DM_ALLOW_RUN_CHANGE=1 / DM_ALLOW_KERNEL_CHANGE=1 accept the change): a configuration error, which a chain of jobs does
not retry. Every rank then prints "FATAL: RESUME REFUSED ..." (REFUSED_MARKER) on its stderr, which the master tees
into <log_root_dir>/<run_name>.log: torchrun reports any worker failure as 1, so a launcher maps the marker back to 2
the way it maps "Stopping early" to 75. Anything else (1, ...) = a failure.
Env: DM_DIST_CLOSE_TIMEOUT (30 s): the bound on the process-group teardown after a stop or a refusal (a healthy one
takes well under a second). DM_WANDB_FINISH_TIMEOUT (60 s, trainer/base_trainer.py) bounds the master's W&B finish that
follows it. A stop costs every rank the rest of the step + the save, and the master up to DM_WANDB_FINISH_TIMEOUT more.
After a SIGTERM that can exceed the 90 s train_arm.sh's collector grants (DM_TERM_GRACE; Slurm's KillWait is 120 s)
on the large arms. A master killed then, in its W&B upload, has written its checkpoint already. With more than one
rank, the others' 75 is the launch's exit status, and the W&B data stays in WANDB_DIR for `wandb sync`. The chains stop
on their time budget (DM_STOP_BEFORE, 600 s before the job's end), not on the SIGTERM. The trainer's own knobs are in
trainer/base_trainer.py's docstring.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import traceback

import torch

from .config import load_config
from .parallel.dist import dist_init, dist_close, dist_barrier, is_master, get_rank, get_world_size
from .utils.misc import seed_all, count_parameters, human_count
from .utils.logging import setup_logging, make_main_log, tee_stdio, BLUE, GREEN, RESET
from .data.data_loader import build_train_dataloader
from .modeling import build_model, FLASH_AVAILABLE, resolve_attn_impl
from .parallel.wrap import build_ddp_model, build_fsdp2_model
from .trainer import DDPTrainer, FSDP2Trainer
from .trainer.base_trainer import ResumeRefused, finish_wandb_bounded

# The fixed text of the line every rank prints when the resume is refused (exit 2); see the module docstring.
REFUSED_MARKER = "RESUME REFUSED"


def main() -> None:
    cfg = load_config()
    _refuse_ddp_with_delta_matching(cfg)   # before any process group, data or model work
    device = dist_init()
    cfg.derive(get_world_size())

    # Tee the master's full stdout+stderr to logs/{run_name}.log (terminal output on disk)
    if is_master():
        tee_stdio(os.path.join(str(cfg.log_root_dir), f"{cfg.run_name}.log"))

    logger = setup_logging(cfg.log_path, rank=get_rank(), log_to_file=True)
    main_LOG = make_main_log(logger, is_master())

    # identical model-init RNG across ranks; data seed is separate (cfg.data.seed)
    seed_all(cfg.seed)

    main_LOG(f"{BLUE}run={cfg.run_name} backend={cfg.backend} world={cfg.num_devices} "
             f"global_batch={cfg.global_batch_size} grad_accum={cfg.gradient_accumulation_steps} "
             f"chunk={cfg.chunk_size}{RESET}", "critical")

    # ---- data ----
    train_loader = build_train_dataloader(
        data_cfg=cfg.data_cfg,
        max_seq_len=cfg.chunk_size,
        global_batch_size=cfg.global_batch_size,
        total_size=cfg.total_size,
        seed=cfg.data.seed,
    )

    # ---- model ----
    cfg.model.attn_impl = resolve_attn_impl(cfg.model.attn_impl)
    main_LOG(f"attention backend: {cfg.model.attn_impl} (flash_attn importable={FLASH_AVAILABLE})")
    model = build_model(cfg.model)
    if cfg.init_from_hf:   # warm-start (continued pretraining): before FSDP wrap, fresh optimizer
        _load_init_weights(model, cfg.init_from_hf, main_LOG)
    if cfg.fused_ce:
        used = model.enable_fused_ce(True)
        main_LOG(f"fused linear cross-entropy: {'ON (liger)' if used else 'requested but unavailable -> unfused'}")
    total, trainable = count_parameters(model)
    analytic = cfg.model.num_params_analytic()   # None when a layer is recurrent (no closed form)
    check = (f"analytic={human_count(analytic)} (match={total == analytic})" if analytic is not None
             else "analytic=n/a (recurrent layers)")
    main_LOG(f"{GREEN}params total={human_count(total)} trainable={human_count(trainable)} {check}{RESET}",
             "critical")
    if cfg.compile:
        main_LOG(f"torch.compile: ON mode={cfg.compile_mode} (regional/per-block; first step compiles)")
    if cfg.fp8:
        main_LOG(f"FP8 training: ON (torchao float8, recipe={cfg.fp8_recipe}, scope={cfg.fp8_scope})")

    # ---- wrap ----
    if cfg.backend == "fsdp2":
        param_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        model = build_fsdp2_model(model, device, activation_checkpointing=cfg.gradient_checkpointing,
                                  ac_mode=cfg.ac_mode, compile=cfg.compile, compile_mode=cfg.compile_mode,
                                  fp8=cfg.fp8, fp8_recipe=cfg.fp8_recipe, fp8_scope=cfg.fp8_scope,
                                  compile_mlp=cfg.compile_mlp, param_dtype=param_dtype)
        Trainer = FSDP2Trainer
    elif cfg.backend == "ddp":
        model = build_ddp_model(model, device, activation_checkpointing=cfg.gradient_checkpointing,
                                ac_mode=cfg.ac_mode, compile=cfg.compile, compile_mode=cfg.compile_mode,
                                fp8=cfg.fp8, fp8_recipe=cfg.fp8_recipe, fp8_scope=cfg.fp8_scope)
        Trainer = DDPTrainer
    else:
        raise ValueError(f"unknown backend {cfg.backend!r}")
    dist_barrier()

    # The Trainer decides, checks and loads the resume. W&B comes after it, so a refused resume (or a load that fails)
    # never resumes the run's W&B id with a configuration that is not trained, nor leaves a crashed segment there. (W&B
    # installs no SIGTERM / SIGUSR1 handler: the trainer's, set in its constructor, stay in place.)
    try:
        trainer = Trainer(model=model, cfg=cfg, train_loader=train_loader, device=device, main_LOG=main_LOG)
    except ResumeRefused as e:
        # a resume no retry can fix (the run identity or the kernel build differs): raised on every rank alike
        print(f"[rank {get_rank()}] FATAL: {REFUSED_MARKER} (exit 2: a configuration error, not retried): {e}",
              file=sys.stderr, flush=True)
        _close_and_exit("refused")
        return
    trainer.wandb_run = _maybe_init_wandb(cfg)
    status = trainer.train()
    _close_and_exit(status, wandb_run=trainer.wandb_run, log=main_LOG)


def _stderr_log(msg, level: str = "info") -> None:
    print(msg, file=sys.stderr, flush=True)


def _close_and_exit(status: str, wandb_run=None, log=None) -> None:
    """The end of a training process: the process group is closed first, then the master's W&B run (`wandb_run`, None
    on the other ranks), then the process exits. W&B comes last because its finish() uploads what is left, up to
    DM_WANDB_FINISH_TIMEOUT (60 s), and only the master has a run. Finished first, it held the other ranks in
    dist_close's barrier, and a slow but normal upload made every rank's bounded teardown time out.
    "stopped" exits 75 (EX_TEMPFAIL: a preemption-safe stop, the newest complete checkpoint holds the state; relaunch to
    resume; the W&B run is marked preempting) and "refused" exits 2 (a resume refused for a reason no retry can fix:
    ResumeRefused), whatever the teardown does: dist_close() runs in a daemon thread joined for DM_DIST_CLOSE_TIMEOUT
    seconds (default 30; a healthy teardown takes well under a second). When it raises (a peer already gone, a broken
    group) or hangs, that is logged, and the process exits with os._exit (after the W&B finish and flushing the logs),
    without the interpreter's exit hooks that could block on the same group. A W&B finish that does not return in time
    takes the os._exit path as well, so W&B's own exit hooks cannot hold the process either. "complete" closes the
    group without a bound; a failure there is an error (exit 1, after the W&B finish), else the process exits 0.
    `log(msg, level)` reports W&B problems (default: stderr)."""
    code = {"stopped": 75, "refused": 2}.get(status, 0)
    hard, close_error = False, None
    if code:
        try:
            timeout = float(os.environ.get("DM_DIST_CLOSE_TIMEOUT", "30") or 30)
        except ValueError:
            timeout = 30.0
        err: list = []

        def close():
            try:
                dist_close()
            except BaseException as e:  # noqa: BLE001 (reported below)
                err.append(e)

        t = threading.Thread(target=close, name="dist-close", daemon=True)
        t.start()
        t.join(max(timeout, 0.0))
        problem = (f"did not return within {timeout:g} s (DM_DIST_CLOSE_TIMEOUT)" if t.is_alive()
                   else f"failed ({type(err[0]).__name__}: {err[0]})" if err else None)
        if problem:
            msg = (f"[rank {get_rank()}] the process-group teardown (dist_close) {problem}; exiting {code} anyway"
                   + (": the stop is complete (the newest complete checkpoint holds the state)" if code == 75 else ""))
            try:
                print(msg, file=sys.stderr, flush=True)    # every rank (the master's stderr is teed to its log)
            except Exception:  # noqa: BLE001
                pass
            hard = True
    else:
        try:
            dist_close()
        except BaseException as e:  # noqa: BLE001 (raised again below, after the W&B finish)
            close_error = e
    # the master's W&B run, now that no other rank waits for this one (bounded; marked preempting on a stop)
    if not finish_wandb_bounded(wandb_run, log or _stderr_log, preempting=(status == "stopped")):
        hard = True
    if close_error is not None:
        if not hard:
            raise close_error
        traceback.print_exception(close_error)
        code = 1
    logging.shutdown()
    if hard:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass
        os._exit(code)
    if code:
        sys.exit(code)


def _refuse_ddp_with_delta_matching(cfg) -> None:
    """The Delta-Matching fp8 attention (the dm mixer with its fused producers, the mla_dm mixer, and the gqa mixer's
    attn_impl='dm') runs on the bf16 activations of the FSDP2 mixed-precision policy (bf16 parameters, so the residual
    stream and the RoPE tables are bf16). DDP keeps fp32 parameters under autocast, so those tensors stay fp32 and the
    first forward fails inside the route. Refuse the combination at startup instead."""
    if cfg.backend != "ddp":
        return
    m = cfg.model
    used = sorted({m.mixer_for_layer(i) for i in range(m.n_layers)} & {"dm", "mla_dm"})
    if m.attn_impl == "dm":
        used.append("attn_impl=dm")
    if used:
        raise ValueError(f"--backend ddp cannot run the Delta-Matching fp8 attention ({', '.join(used)}): the route "
                         "needs the bf16 activations of the FSDP2 mixed-precision policy, and DDP keeps them in fp32. "
                         "Use --backend fsdp2 (it also runs on a single GPU).")


def _load_init_weights(model, hf_dir, main_LOG):
    """Warm-start: load consolidated weights from a finished run's HF export (model.safetensors, src/pretrain
    naming: see hf/export.py) into the unwrapped training model. Fresh optimizer/schedule (the trainer
    starts at step 0). Tied lm_head.weight is dropped in the export and re-tied here -> the only allowed
    missing key. Loads on CPU then the FSDP wrap shards it."""
    from safetensors.torch import load_file
    path = os.path.join(str(hf_dir), "model.safetensors")
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]   # tied -> expected
    if missing or unexpected:
        raise RuntimeError(f"init_from_hf mismatch: missing={missing} unexpected={unexpected}")
    main_LOG(f"warm-start: loaded {len(sd)} tensors from {path} (init_from_hf; fresh optimizer/schedule)", "critical")


def _maybe_init_wandb(cfg):
    """Master-rank-only wandb init. Returns the run (or None). Never raises: a wandb
    failure must not crash training."""
    if not (cfg.wandb and is_master()):
        return None
    try:
        import dataclasses
        import wandb

        # --wandb true is an explicit opt-in: it overrides an inherited WANDB_MODE=disabled
        # (e.g. from the shell environment) with online, but respects an explicit `offline`.
        if os.environ.get("WANDB_MODE") == "disabled":
            os.environ["WANDB_MODE"] = "online"

        # The wandb dir follows the workspace (WANDB_DIR, else the parent of log_root_dir), and a relaunched run
        # RESUMES its wandb run through the id stored beside
        # the logs, so a preempted training keeps one curve instead of one wandb run per attempt.
        wdir = os.environ.get("WANDB_DIR") or os.path.dirname(str(cfg.log_root_dir)) or "."
        id_path = os.path.join(str(cfg.log_root_dir), f"{cfg.run_name}.wandb_id")
        kw = {}
        if os.path.isfile(id_path):
            kw = {"id": open(id_path).read().strip(), "resume": "allow"}
        run = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project,
                         name=cfg.run_name, job_type=cfg.wandb_job_type,
                         config=dataclasses.asdict(cfg), dir=wdir, **kw)
        if not os.path.isfile(id_path):
            try:
                with open(id_path, "w") as fh:
                    fh.write(str(run.id))
            except OSError:
                pass
        _append_overview(cfg, run.id)
        return run
    except Exception as e:  # never let logging kill training
        print(f"[wandb] disabled (init failed): {e}", flush=True)
        return None


def _append_overview(cfg, run_id: str) -> None:
    """Append `run_name<TAB>run_id` to logs/wandb_runs.txt (an index of the wandb runs of a
    log root) under an exclusive lock (two runs start in parallel). Best-effort; never raises."""
    try:
        import portalocker

        log_root = str(cfg.log_root_dir)
        os.makedirs(log_root, exist_ok=True)
        path = os.path.join(log_root, "wandb_runs.txt")
        with portalocker.Lock(path, mode="a", timeout=30) as fh:
            fh.write(f"{cfg.run_name}\t{run_id}\n")
    except Exception as e:
        print(f"[wandb] could not write wandb_runs.txt: {e}", flush=True)


if __name__ == "__main__":
    main()
