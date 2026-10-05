"""Engine subclasses of the dmlm mixers: override ONLY the cache interaction.

Produced by rebinding `__class__` onto the loaded dmlm mixers (see dm_model.py), so every
projection / kernel / norm weight is inherited unchanged -> numerically faithful to the HF
`.forward` by construction. `forward(hidden_states, ctx)` returns the mixer output (post
o_proj/out_proj) to add as the residual.

- EngineGQA: RoPE (our exact `apply_rotary`, applied manually so the cache stores rotated K) +
  `flash_attn_with_kvcache` (decode, slot-gathered) / `flash_attn_func` (prefill).
- EngineMLA: the same cache interaction on MLAAttention's assembled per-head q/k/v.
- EngineMamba2 / EngineGDN / EngineKDA: recurrent state, slot-gathered.

EngineMamba2 is adapted from mamba_ssm's `Mamba2.forward` / `Mamba2.step` (mamba_ssm/modules/mamba2.py;
Copyright (c) 2024, Tri Dao, Albert Gu, Apache-2.0). EngineGDN and EngineKDA are adapted from fla's
`GatedDeltaNet.forward` / `KimiDeltaAttention.forward` (fla/layers/gated_deltanet.py, fla/layers/kda.py;
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li, MIT). Changes: the cache interaction only
(per-slot states of dm_cache.py, gathered and scattered by slot id).
"""
from __future__ import annotations

import torch

from flash_attn import flash_attn_func, flash_attn_with_kvcache

from pretrain.modeling.attention import GQAAttention
from pretrain.modeling.mamba2 import Mamba2Mixer
from pretrain.modeling.gated_deltanet import GatedDeltaNetMixer
from pretrain.modeling.kda import KDAMixer
from pretrain.modeling.mla import MLAAttention
from pretrain.modeling.layers import apply_output_gate, rotate_half, split_q_gate

from .context import PHASE_DECODE


def _rope(q, k, cos, sin):
    """q (B,T,H,D), k (B,T,Hkv,D); cos/sin (B,T,D). Same math as
    `pretrain.modeling.layers.apply_rotary` but for the flash (seqlen-second, head-third)
    layout: broadcast cos/sin over the head dim (2) instead of (1)."""
    cos = cos.unsqueeze(2)  # (B,T,1,R)
    sin = sin.unsqueeze(2)
    rd = cos.shape[-1]
    if rd == q.shape[-1]:
        qf, kf = q.float(), k.float()
        q = (qf * cos + rotate_half(qf) * sin).to(q.dtype)
        k = (kf * cos + rotate_half(kf) * sin).to(k.dtype)
        return q, k
    # PARTIAL RoPE (Qwen3.5): rotate the leading rd channels, pass the tail through. Must mirror
    # pretrain/modeling/layers.py::apply_rotary exactly or eval silently scores a different model
    # than the one that was trained: rotate_half has to be taken WITHIN the rotated slice.
    def _sr(x):
        xr, xp = x[..., :rd].float(), x[..., rd:]
        return torch.cat([(xr * cos + rotate_half(xr) * sin).to(x.dtype), xp], dim=-1)
    return _sr(q), _sr(k)


class EngineGQA(GQAAttention):
    """Continuous-batching GQA with RoPE + a per-slot KV pool via FlashAttention."""

    def forward(self, hidden_states, ctx):  # type: ignore[override]
        pool = ctx.pool
        i = self._eidx
        B, T, _ = hidden_states.shape
        # (B, T, H, D): seqlen-second layout for flash_attn
        # attn_output_gate (Qwen3.5): q_proj emits 2*q_dim and the second half is the gate.
        # split_q_gate is the SAME helper the training mixers use, so the per-head split
        # convention cannot drift between train and eval.
        q, gate = split_q_gate(self.q_proj(hidden_states), self.n_heads, self.head_dim,
                               getattr(self, "attn_output_gate", False))
        k = self.k_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim)
        if self.qk_norm:                      # per-head RMSNorm over head_dim (last dim), before RoPE
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = _rope(q, k, ctx.cos, ctx.sin)  # rotate; the cache stores rotated K
        if self.attn_mscale != 1.0:           # YaRN temperature: logits x mscale^2 (match the training attention; q-only)
            q = q * (self.attn_mscale * self.attn_mscale)

        if ctx.phase == PHASE_DECODE:
            lens = pool.seqlens.index_select(0, ctx.slot_ids)  # pre-write lengths (= new-token positions)
            out = flash_attn_with_kvcache(
                q, pool.k_pool[i], pool.v_pool[i], k=k, v=v,
                cache_seqlens=lens, cache_batch_idx=ctx.slot_ids,
                softmax_scale=self.scale, causal=True,
            )  # appends (rotated) k,v into the slot in-place, then attends
        else:
            slot = ctx.prefill_slot
            out = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=True)
            pool.k_pool[i][slot, :T].copy_(k[0])
            pool.v_pool[i][slot, :T].copy_(v[0])

        out = out.reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(apply_output_gate(out, gate))


class EngineMLA(MLAAttention):
    """Continuous-batching MLA: a per-slot KV pool via FlashAttention, exactly as EngineGQA.

    MLAAttention._qkv already returns the FINISHED per-head q/k/v: the joint latent expanded to
    per-head nope keys and values, the shared rope key rotated once and broadcast across heads,
    the four sub-vector qk norms, and (stage 2) the YaRN attention temperature on q. Reusing it
    verbatim is what makes this path numerically faithful to the training forward: nothing about
    the MLA math is reimplemented here, only the cache interaction.

    Two shape facts make the GQA pool fit. MLA is MHA on the wire (`n_kv_heads == n_heads`, asserted
    in MLAAttention.__init__), and K is assembled as [rope(rotary_dim) | nope(head_dim - rotary_dim)],
    i.e. head_dim wide, so K and V are [B, T, H, head_dim], the layout k_pool/v_pool already hold.
    We cache the ASSEMBLED K rather than the compressed latent: the latent is smaller, but the
    assembled form is what flash_attn_with_kvcache can append to in place, and at eval batch sizes
    the memory difference is irrelevant. `_qkv` yields head-second [B, H, T, D]; flash wants
    seqlen-second, hence the transpose.

    softmax_scale: MLAAttention's SDPA path calls F.scaled_dot_product_attention WITHOUT an explicit
    scale, so torch applies 1/sqrt(E) with E = head_dim, identical to self.scale (head_dim ** -0.5),
    which is therefore the correct value to hand flash. The YaRN mscale is already folded into q by
    _qkv, so it must NOT be applied again here.
    """

    def forward(self, hidden_states, ctx):  # type: ignore[override]
        pool = ctx.pool
        i = self._eidx
        B, T, _ = hidden_states.shape
        # ctx.cos/ctx.sin are (B, T, rotary_dim) in both phases (prefill 0..T-1, decode the slot's
        # current length), which is precisely what _rope_leading broadcasts over the head dim.
        q, k, v, gate = self._qkv(hidden_states, ctx.cos, ctx.sin)
        q = q.transpose(1, 2).contiguous()          # (B,H,T,D) -> (B,T,H,D)
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if ctx.phase == PHASE_DECODE:
            lens = pool.seqlens.index_select(0, ctx.slot_ids)   # pre-write lengths = new-token positions
            out = flash_attn_with_kvcache(
                q, pool.k_pool[i], pool.v_pool[i], k=k, v=v,
                cache_seqlens=lens, cache_batch_idx=ctx.slot_ids,
                softmax_scale=self.scale, causal=True,
            )  # appends k,v into the slot in-place, then attends
        else:
            slot = ctx.prefill_slot
            out = flash_attn_func(q, k, v, softmax_scale=self.scale, causal=True)
            pool.k_pool[i][slot, :T].copy_(k[0])
            pool.v_pool[i][slot, :T].copy_(v[0])

        out = out.reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(apply_output_gate(out, gate))


class EngineMamba2(Mamba2Mixer):
    """Continuous-batching Mamba2 (wraps mamba_ssm.Mamba2 on self.mamba): pool-backed conv+ssm
    state gathered by slot id. Same kernels as the HF prefill forward (mamba_chunk_scan_combined)
    and the mamba_ssm step (causal_conv1d_update + selective_state_update), so it is numerically
    faithful by construction. Mirrors inference/layers.py::EngineMamba2Mixer with mamba_ssm attrs."""

    def forward(self, hidden_states, ctx):  # type: ignore[override]
        if ctx.phase == PHASE_DECODE:
            return self._decode(hidden_states, ctx)
        return self._prefill(hidden_states, ctx)

    def _decode(self, hidden_states, ctx):
        from causal_conv1d import causal_conv1d_update
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        m = self.mamba
        pool, slot_ids, i = ctx.pool, ctx.slot_ids, self._eidx
        bsz = hidden_states.shape[0]
        d_inner, nheads, ng, ds, hd = m.d_inner, m.nheads, m.ngroups, m.d_state, m.headdim
        conv_dim = m.conv1d.weight.shape[0]
        gtss = ng * ds

        zxbcdt = m.in_proj(hidden_states.squeeze(1))
        z, xBC, dt = torch.split(zxbcdt, [d_inner, conv_dim, nheads], dim=-1)
        xBC = causal_conv1d_update(xBC, pool.conv_pool[i], m.conv1d.weight.squeeze(1),
                                   m.conv1d.bias, m.activation, conv_state_indices=slot_ids)
        x, B, C = torch.split(xBC, [d_inner, gtss, gtss], dim=-1)
        A = (-torch.exp(m.A_log.float()))[:, None, None].expand(-1, hd, ds).to(torch.float32)
        dt = dt[:, :, None].expand(-1, -1, hd)
        dt_bias = m.dt_bias[:, None].expand(-1, hd)
        D = m.D[:, None].expand(-1, hd)
        B = B.view(bsz, ng, ds)
        C = C.view(bsz, ng, ds)
        x = x.view(bsz, nheads, hd)
        y = selective_state_update(pool.ssm_pool[i], x, dt, A, B, C, D, z=None,
                                   dt_bias=dt_bias, dt_softplus=True, state_batch_indices=slot_ids)
        y = y.view(bsz, nheads * hd)
        y = m.norm(y, z)
        return m.out_proj(y)[:, None]

    def _prefill(self, hidden_states, ctx):
        from causal_conv1d import causal_conv1d_fn
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        m = self.mamba
        pool, slot, i = ctx.pool, ctx.prefill_slot, self._eidx
        bsz, L, _ = hidden_states.shape
        d_inner, nheads, ng, ds, hd = m.d_inner, m.nheads, m.ngroups, m.d_state, m.headdim
        conv_dim = m.conv1d.weight.shape[0]
        gtss = ng * ds

        zxbcdt = m.in_proj(hidden_states)
        z, xBC, dt = torch.split(zxbcdt, [d_inner, conv_dim, nheads], dim=-1)
        # conv state = last d_conv cols of the pre-conv xBC (left zero-padded)
        xBC_t = xBC.transpose(1, 2)  # (1, conv_dim, L)
        conv_state = torch.nn.functional.pad(xBC_t, (m.d_conv - xBC_t.shape[-1], 0))
        pool.conv_pool[i][slot].copy_(conv_state[0])
        xBC = causal_conv1d_fn(xBC_t, m.conv1d.weight.squeeze(1), m.conv1d.bias,
                               activation=m.activation).transpose(1, 2)[:, :L]
        x, B, C = torch.split(xBC, [d_inner, gtss, gtss], dim=-1)
        A = -torch.exp(m.A_log.float())
        y, ssm_state = mamba_chunk_scan_combined(
            x.view(bsz, L, nheads, hd), dt, A,
            B.view(bsz, L, ng, ds), C.view(bsz, L, ng, ds),
            chunk_size=m.chunk_size, D=m.D, z=None, seq_idx=None,
            return_final_states=True, dt_bias=m.dt_bias, dt_softplus=True)
        pool.ssm_pool[i][slot].copy_(ssm_state[0])
        y = y.view(bsz, L, nheads * hd)
        y = m.norm(y, z)
        return m.out_proj(y)


class EngineGDN(GatedDeltaNetMixer):
    """Continuous-batching Gated DeltaNet (wraps fla.layers.GatedDeltaNet on self.gdn).

    Reuses fla's exact building blocks (the q/k/v ShortConvolutions with conv `cache=` and
    chunk_/fused_recurrent_gated_delta_rule with `initial_state=`), so it is numerically
    faithful to the HF forward. Per-slot recurrent (fp32) + 3 conv states are gathered/scattered
    by slot id (fla's recurrent kernel is batch-, not index-, gathered)."""

    def _qkv(self, g, h, cq, ck, cv, out_final):
        from einops import rearrange
        q, cq = g.q_conv1d(x=g.q_proj(h), cache=cq, output_final_state=out_final, cu_seqlens=None)
        k, ck = g.k_conv1d(x=g.k_proj(h), cache=ck, output_final_state=out_final, cu_seqlens=None)
        v, cv = g.v_conv1d(x=g.v_proj(h), cache=cv, output_final_state=out_final, cu_seqlens=None)
        q = rearrange(q, '... (hh d) -> ... hh d', d=g.head_k_dim)
        k = rearrange(k, '... (hh d) -> ... hh d', d=g.head_k_dim)
        v = rearrange(v, '... (hh d) -> ... hh d', d=g.head_v_dim)
        return q, k, v, cq, ck, cv

    def _out(self, g, h, o):
        from einops import rearrange
        if g.use_gate:
            gg = rearrange(g.g_proj(h), '... (hh d) -> ... hh d', d=g.head_v_dim)
            o = g.o_norm(o, gg)
        else:
            o = g.o_norm(o)
        return g.o_proj(rearrange(o, 'b t hh d -> b t (hh d)'))

    def forward(self, hidden_states, ctx):  # type: ignore[override]
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        g = self.gdn
        pool, i = ctx.pool, self._eidx
        beta = g.b_proj(hidden_states).sigmoid()
        if g.allow_neg_eigval:
            beta = beta * 2.0
        a = g.a_proj(hidden_states)

        if ctx.phase == PHASE_DECODE:
            sid = ctx.slot_ids.long()   # index_copy_/index_select need int64
            cq = pool.gconv_q[i].index_select(0, sid).contiguous()
            ck = pool.gconv_k[i].index_select(0, sid).contiguous()
            cv = pool.gconv_v[i].index_select(0, sid).contiguous()
            rec = pool.recur_pool[i].index_select(0, sid).contiguous()
            q, k, v, cq, ck, cv = self._qkv(g, hidden_states, cq, ck, cv, True)
            o, rec = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v, g=a, beta=beta, initial_state=rec, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=g.A_log, dt_bias=g.dt_bias)
            pool.recur_pool[i].index_copy_(0, sid, rec)
            pool.gconv_q[i].index_copy_(0, sid, cq)
            pool.gconv_k[i].index_copy_(0, sid, ck)
            pool.gconv_v[i].index_copy_(0, sid, cv)
        else:
            slot = ctx.prefill_slot
            q, k, v, cq, ck, cv = self._qkv(g, hidden_states, None, None, None, True)
            o, rec = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=a, beta=beta, initial_state=None, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=g.A_log, dt_bias=g.dt_bias)
            pool.recur_pool[i][slot].copy_(rec[0])
            pool.gconv_q[i][slot].copy_(cq[0])
            pool.gconv_k[i][slot].copy_(ck[0])
            pool.gconv_v[i][slot].copy_(cv[0])
        return self._out(g, hidden_states, o)


class EngineKDA(KDAMixer):
    """Continuous-batching Kimi Delta Attention (wraps fla.layers.kda.KimiDeltaAttention on self.kda).

    The EngineGDN idiom on fla's KDA building blocks: the q/k/v ShortConvolutions with conv `cache=`, and
    chunk_kda / fused_recurrent_kda with `initial_state=`, the per-channel decay `g = f_proj(h)` (H x K) and
    `beta = sigmoid(b_proj(h))` formed exactly as the layer's own forward forms them (use_qk_l2norm_in_kernel and
    use_gate_in_kernel on, safe_gate / lower_bound from the layer), the gated o_norm + o_proj epilogue. Per-slot
    fp32 recurrent state + 3 conv states gathered/scattered by slot id."""

    def _qkv(self, g, h, cq, ck, cv, out_final):
        from einops import rearrange
        q, cq = g.q_conv1d(x=g.q_proj(h), cache=cq, output_final_state=out_final, cu_seqlens=None)
        k, ck = g.k_conv1d(x=g.k_proj(h), cache=ck, output_final_state=out_final, cu_seqlens=None)
        v, cv = g.v_conv1d(x=g.v_proj(h), cache=cv, output_final_state=out_final, cu_seqlens=None)
        q = rearrange(q, '... (hh d) -> ... hh d', d=g.head_k_dim)
        k = rearrange(k, '... (hh d) -> ... hh d', d=g.head_k_dim)
        v = rearrange(v, '... (hh d) -> ... hh d', d=g.head_v_dim)
        return q, k, v, cq, ck, cv

    def _out(self, g, h, o):
        from einops import rearrange
        o = g.o_norm(o, rearrange(g.g_proj(h), '... (hh d) -> ... hh d', d=g.head_v_dim))
        return g.o_proj(rearrange(o, 'b t hh d -> b t (hh d)'))

    def forward(self, hidden_states, ctx):  # type: ignore[override]
        from einops import rearrange
        from fla.ops.kda import chunk_kda, fused_recurrent_kda
        g = self.kda
        pool, i = ctx.pool, self._eidx
        beta = g.b_proj(hidden_states).sigmoid()
        if g.allow_neg_eigval:
            beta = beta * 2.0
        gate = rearrange(g.f_proj(hidden_states), '... (hh d) -> ... hh d', d=g.head_k_dim)

        if ctx.phase == PHASE_DECODE:
            sid = ctx.slot_ids.long()
            cq = pool.gconv_q[i].index_select(0, sid).contiguous()
            ck = pool.gconv_k[i].index_select(0, sid).contiguous()
            cv = pool.gconv_v[i].index_select(0, sid).contiguous()
            rec = pool.recur_pool[i].index_select(0, sid).contiguous()
            q, k, v, cq, ck, cv = self._qkv(g, hidden_states, cq, ck, cv, True)
            o, rec = fused_recurrent_kda(
                q=q, k=k, v=v, g=gate, beta=beta, A_log=g.A_log, dt_bias=g.dt_bias, initial_state=rec,
                output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, lower_bound=g.lower_bound)
            pool.recur_pool[i].index_copy_(0, sid, rec)
            pool.gconv_q[i].index_copy_(0, sid, cq)
            pool.gconv_k[i].index_copy_(0, sid, ck)
            pool.gconv_v[i].index_copy_(0, sid, cv)
        else:
            slot = ctx.prefill_slot
            q, k, v, cq, ck, cv = self._qkv(g, hidden_states, None, None, None, True)
            o, rec = chunk_kda(
                q=q, k=k, v=v, g=gate, beta=beta, A_log=g.A_log, dt_bias=g.dt_bias, initial_state=None,
                output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                safe_gate=g.safe_gate, lower_bound=g.lower_bound)
            pool.recur_pool[i][slot].copy_(rec[0])
            pool.gconv_q[i][slot].copy_(cq[0])
            pool.gconv_k[i][slot].copy_(ck[0])
            pool.gconv_v[i][slot].copy_(cv[0])
        return self._out(g, hidden_states, o)
