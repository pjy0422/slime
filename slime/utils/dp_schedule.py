"""Per-rollout DP/microbatch scheduling.

Pure-Python logic that decides, for one rollout's worth of sample lengths,
how to group samples into micro-batches and which DP rank owns each mbs.
Lives outside the ray/sglang-importing modules so it can be unit-tested
under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. Group samples by rollout id (``rollout_indices[i]`` =
     ``samples[i].index``) and split rollouts into steps of
     ``global_batch_size`` rollouts each. In the common case one rollout
     emits one training sample so this is the same as a contiguous chunk;
     under compact / subagent one rollout may emit multiple training
     samples, in which case all of those samples stay in the same step.
  2. For each step, pack its samples into ``K`` micro-batches with a
     single first-fit pass (dynamic batch) or fixed-size chunking
     (static batch).
  3. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only).
  4. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
     each, with either a strided round-robin or a Karmarkar-Karp pass on
     estimated mbs FLOPs.

With ``--rollout-dp-affinity``, steps are still formed the same way, but whole
rollout bundles are assigned to ranks before rank-local packing. Dynamic bins
may be split further to equalize the per-rank micro-batch count; the scheduler
fails instead of moving part of a rollout to another rank when that is not
possible.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path without ``balance_by_flops``) holds
    ``<= max_tokens_per_gpu * cp_size`` tokens, with one exception — an
    individual sample larger than that cap lands alone in its own mbs (and
    that mbs is the only one allowed to exceed the cap);
  - the union of per-rank sample indices equals the set of samples kept
    after trimming trailing rollouts (every kept sample placed exactly
    once);
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)`` (each rank's samples are tiled exactly
    once by its mbs schedule).
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.flops_utils import calculate_fwd_flops
from slime.utils.seqlen_balancing import expand_bins_by_splitting, first_fit_pack, get_seqlen_balanced_partitions

logger = logging.getLogger(__name__)

DP_LOCAL_DATA_KEYS = (
    "tokens",
    "multimodal_train_inputs",
    "response_lengths",
    "rewards",
    "truncated",
    "loss_masks",
    "round_number",
    "sample_indices",
    "group_indices",
    "rollout_ids",
    "rollout_mask_sums",
    "rollout_log_probs",
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
    "rollout_routed_experts",
    "source_names",
    "prompt",
    "teacher_log_probs",
    "metadata",
)
DP_GLOBAL_DATA_KEYS = ("raw_reward", "total_lengths")


def _calculate_workloads(step_lengths, args):
    return [calculate_fwd_flops([sl], args) for sl in step_lengths]


def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    args: Any,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
    balance_by_flops: bool = False,
) -> list[list[int]]:
    """Group a step's samples into mbs. Returns ``mbs[k]`` = local indices into ``step_lengths``."""
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        if balance_by_flops:
            total_tokens = sum(step_lengths)
            num_mbs = max(1, (total_tokens + max_per_bin - 1) // max_per_bin)
            if num_mbs >= len(step_lengths):
                return [[i] for i in range(len(step_lengths))]
            workloads = _calculate_workloads(step_lengths, args)
            # FLOPs balancing does not enforce the token cap per mbs. A
            # partition can exceed max_per_bin and may OOM if the cap is tight.
            return get_seqlen_balanced_partitions(workloads, num_mbs, equal_size=False)
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]


def partition_train_data(data: dict[str, Any], partition: list[int]) -> dict[str, Any]:
    """Select one DP rank's fields while retaining required global fields."""

    rollout_data = {key: [data[key][index] for index in partition] for key in DP_LOCAL_DATA_KEYS if key in data}
    rollout_data.update({key: data[key] for key in DP_GLOBAL_DATA_KEYS if key in data})
    return rollout_data


def _assign_rollout_bundles(
    rollout_ids: list[int],
    rollout_id_to_samples: dict[int, list[int]],
    total_lengths: list[int],
    *,
    args: Any,
    dp_size: int,
) -> list[list[int]]:
    """Assign complete rollout bundles to ranks with deterministic load balancing."""

    if len(rollout_ids) < dp_size:
        raise ValueError(
            "rollout DP affinity requires at least one rollout per DP rank in every training step; "
            f"got {len(rollout_ids)} rollouts for dp_size={dp_size}"
        )

    sample_workloads = _calculate_workloads(total_lengths, args) if args.balance_data else total_lengths
    rollout_order = {rollout_id: index for index, rollout_id in enumerate(rollout_ids)}
    bundles = [
        (
            rollout_id,
            rollout_id_to_samples[rollout_id],
            sum(sample_workloads[index] for index in rollout_id_to_samples[rollout_id]),
        )
        for rollout_id in rollout_ids
    ]
    if args.balance_data:
        bundles.sort(key=lambda item: (-item[2], -len(item[1]), rollout_order[item[0]]))
    else:
        bundles.sort(key=lambda item: (-len(item[1]), -item[2], rollout_order[item[0]]))

    rank_rollouts: list[list[int]] = [[] for _ in range(dp_size)]
    rank_sample_counts = [0] * dp_size
    rank_workloads = [0] * dp_size
    for rollout_id, sample_indices, workload in bundles:
        rank = min(
            range(dp_size),
            key=lambda candidate: (
                rank_workloads[candidate] if args.balance_data else rank_sample_counts[candidate],
                rank_sample_counts[candidate],
                len(rank_rollouts[candidate]),
                candidate,
            ),
        )
        rank_rollouts[rank].append(rollout_id)
        rank_sample_counts[rank] += len(sample_indices)
        rank_workloads[rank] += workload

    # Restore first-occurrence rollout order within each rank. Assignment is
    # load-aware, but the temporal/data-source order remains deterministic.
    for assigned in rank_rollouts:
        assigned.sort(key=rollout_order.__getitem__)
    return rank_rollouts


def _pack_step_with_rollout_affinity(
    step_rollouts: list[int],
    rollout_id_to_samples: dict[int, list[int]],
    total_lengths: list[int],
    *,
    args: Any,
    dp_size: int,
    max_per_bin: int | None,
    mb_group: int,
) -> list[list[list[int]]]:
    """Pack one step while keeping every rollout on exactly one DP rank."""

    rank_rollouts = _assign_rollout_bundles(
        step_rollouts,
        rollout_id_to_samples,
        total_lengths,
        args=args,
        dp_size=dp_size,
    )
    rank_samples = [
        [sample for rollout_id in assigned for sample in rollout_id_to_samples[rollout_id]]
        for assigned in rank_rollouts
    ]
    rank_mbs: list[list[list[int]]] = []
    for sample_indices in rank_samples:
        local_lengths = [total_lengths[index] for index in sample_indices]
        local_mbs = _pack_step_into_mbs(
            local_lengths,
            args=args,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
            balance_by_flops=args.balance_by_flops,
        )
        rank_mbs.append([[sample_indices[local] for local in mbs] for mbs in local_mbs])

    rank_mbs_counts = [len(mbs) for mbs in rank_mbs]
    target_mbs = max(((count + mb_group - 1) // mb_group) * mb_group for count in rank_mbs_counts)
    if not args.use_dynamic_batch_size:
        if any(count != target_mbs for count in rank_mbs_counts):
            raise ValueError(
                "rollout DP affinity cannot produce equal static micro-batch counts without splitting a fixed "
                f"micro-batch; per-rank counts={rank_mbs_counts}, required={target_mbs}"
            )
        return rank_mbs

    for rank, mbs in enumerate(rank_mbs):
        if len(mbs) == target_mbs:
            continue
        local_lengths = [total_lengths[index] for index in rank_samples[rank]]
        local_positions = {sample_index: position for position, sample_index in enumerate(rank_samples[rank])}
        local_mbs = [[local_positions[index] for index in bin_] for bin_ in mbs]
        expand_bins_by_splitting(local_mbs, target_mbs, local_lengths)
        if len(local_mbs) != target_mbs:
            raise ValueError(
                "rollout DP affinity cannot produce equal micro-batch counts without moving a rollout across "
                f"ranks; rank={rank}, samples={len(rank_samples[rank])}, produced={len(local_mbs)}, "
                f"required={target_mbs}"
            )
        rank_mbs[rank] = [[rank_samples[rank][local] for local in bin_] for bin_ in local_mbs]
    return rank_mbs


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    rollout_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``, and optional
            ``rollout_dp_affinity``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, indexed globally.
        global_batch_size: number of rollouts (NOT training samples) per
            training step. Number of training steps =
            ``num_rollouts // global_batch_size``; trailing rollouts whose
            samples don't fit are dropped.
        rollout_indices: rollout id for each sample (``samples[i].index``).
            Samples sharing the same id are kept together in one step.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
        ``global_batch_sizes[s]`` = rollout count for step s (constant
        ``global_batch_size`` for every step).
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # mbs count per step must be divisible by (dp_size * mb_group_for_vpp) so
    # every rank ends up with the same num_mbs and (for VPP) the per-rank mbs
    # count is a multiple of mb_group.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    # Group samples by rollout id (preserve first-occurrence order). All
    # samples from one rollout stay in a single step so the per-rollout loss
    # reducer is well-defined.
    rollout_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, rid in enumerate(rollout_indices):
        rollout_id_to_samples.setdefault(rid, []).append(sample_pos)
    rollout_ids = list(rollout_id_to_samples.keys())

    num_steps = len(rollout_ids) // global_batch_size
    assert num_steps >= 1, (
        f"num_rollouts ({len(rollout_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one rollout per step."
    )

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_i in range(num_steps):
        step_rollouts = rollout_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
        sample_indices = [pos for rid in step_rollouts for pos in rollout_id_to_samples[rid]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        global_batch_sizes.append(global_batch_size)
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        if getattr(args, "rollout_dp_affinity", False):
            rank_step_mbs = _pack_step_with_rollout_affinity(
                step_rollouts,
                rollout_id_to_samples,
                total_lengths,
                args=args,
                dp_size=dp_size,
                max_per_bin=max_per_bin,
                mb_group=mb_group if vpp_size > 1 else 1,
            )
            step_num_microbatches = len(rank_step_mbs[0])
            assert all(len(mbs) == step_num_microbatches for mbs in rank_step_mbs)
            num_microbatches.append(step_num_microbatches)
            for rank, rank_mbs in enumerate(rank_step_mbs):
                for mbs in rank_mbs:
                    local_start = len(partitions[rank])
                    partitions[rank].extend(mbs)
                    micro_batch_indices[rank].append(list(range(local_start, local_start + len(mbs))))
            continue

        # 1. Pack samples in this step into mbs with one global pass.
        # ``step_mbs`` indices are LOCAL into ``sample_indices``.
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            args=args,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
            balance_by_flops=args.balance_by_flops,
        )

        # 2. Align mbs count to a multiple of ``align_to``.
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                assert len(step_mbs) == target_K, (
                    f"dynamic path: could only produce {len(step_mbs)} mbs after maximal splitting; "
                    f"need {target_K}. step {step_i} has {len(sample_indices)} samples, below the "
                    f"alignment threshold ({align_to})."
                )
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        num_microbatches.append(num_mbs_per_rank)

        # 3. Distribute mbs across ranks: KK on estimated FLOPs when rank
        # workload balancing is enabled, otherwise a strided round-robin.
        if args.balance_data:
            step_workloads = _calculate_workloads(step_lengths, args)
            mbs_weights = [sum(step_workloads[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_weights, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

        # 4. Build per-rank partitions (global sample indices) and micro_batch_indices
        # (local indices into partitions[r]).
        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]  # local indices into sample_indices
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes
