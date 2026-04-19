"""Variable-tracking task generation for multi-hop dependency chains."""

from __future__ import annotations

from ..helpers import make_rng, random_number, random_var
from ..models import PrefillSegment, RelevantSpan, TaskExample
from .common import search_context


# This generator builds variable-tracking tasks by hiding a chain of assignments
# inside long filler text, then asks the model to recover the final numeric value.
# The chain placement (depths and tail anchoring) lets evaluators probe memory
# eviction and multi-hop reasoning stability across context lengths.


def build_vt_example(
    task_name: str,
    index: int,
    target_context_length: int,
    tokenizer,
    *,
    num_hops: int,
    depths: list[float],
    terminal_depth: float | None,
) -> TaskExample:
    """Build a variable-assignment chain hidden inside long filler text."""
    if len(depths) > num_hops:
        raise ValueError(f"Expected at most {num_hops} hop depths for {task_name}; got {len(depths)}.")

    # --- Chain setup: deterministic variable and value selection per sample. ---
    # Every sample gets a deterministic chain like VAR_A = VAR_B = ... = 12345
    # so we can later pinpoint exactly which hop was lost.
    rng = make_rng((target_context_length * 2000) + index + num_hops)
    variables = [random_var(rng) for _ in range(num_hops + 1)]
    final_value = random_number(rng)
    chain_statements = [f"{variables[hop_index]} = {variables[hop_index + 1]}." for hop_index in range(num_hops)]
    terminal_statement = f"{variables[-1]} = {final_value}."

    # --- Depth placement: split hops into body vs tail inserts based on depths. ---
    body_specs: list[tuple[str, str, str, float, dict]] = []
    tail_specs: list[tuple[str, str, str, float, dict]] = []
    for hop_index, statement in enumerate(chain_statements, start=1):
        # Earlier hops can be pinned to specific depths; any leftover hops get
        # shoved to the tail to create especially eviction-sensitive links.
        target_specs = body_specs if hop_index <= len(depths) else tail_specs
        target_specs.append(
            (
                f"hop_{hop_index}",
                "hop",
                statement,
                depths[hop_index - 1] if hop_index <= len(depths) else 1.0,
                {"statement": statement, "hop_index": hop_index},
            )
        )

    terminal_spec = (
        "terminal_value",
        "terminal",
        terminal_statement,
        1.0 if terminal_depth is None else terminal_depth,
        {"statement": terminal_statement, "value": final_value},
    )
    if terminal_depth is None:
        tail_specs.append(terminal_spec)
    else:
        body_specs.append(terminal_spec)

    # --- Context construction: embed statements at target depths in filler text. ---
    # Build a long essay-style context that hides each assignment statement at
    # the requested depth fractions.
    prefix = "Memorize and follow the variable assignment chain hidden in the following text.\n\n"
    built = search_context(
        tokenizer=tokenizer,
        target_context_length=target_context_length,
        prefix=prefix,
        base_filler_kind="essay",
        inserts=[spec[2] for spec in body_specs],
        depths=[spec[3] for spec in body_specs],
        tail_inserts=[spec[2] for spec in tail_specs],
    )

    relevant_spans = []
    for (name, kind, _, depth_fraction, metadata), (char_start, char_end) in zip(body_specs + tail_specs, built.insert_spans):
        # Each hop becomes a traceable span so the evaluator can identify the
        # first place where the reasoning chain was broken by eviction.
        relevant_spans.append(
            RelevantSpan(
                name=name,
                kind=kind,
                char_start=char_start,
                char_end=char_end,
                depth_fraction=depth_fraction,
                metadata=metadata,
            )
        )

    query_var = variables[0]
    question = f"What is the final numeric value of {query_var}?"
    answer_prefix = f" Answer: The final numeric value of {query_var} is "

    # --- Final task construction: package context, question, and labels. ---
    return TaskExample(
        index=index,
        task_name=task_name,
        task_family="vt",
        context=built.context,
        question=question,
        answer_prefix=answer_prefix,
        outputs=[final_value],
        max_new_tokens=24,
        target_context_length=target_context_length,
        relevant_spans=relevant_spans,
        prefill_segments=[PrefillSegment(name="context", char_start=0, char_end=len(built.context))],
        metadata={
            "query_var": query_var,
            "variables": variables,
            # These fields make the later error taxonomy explainable instead of
            # just reporting "wrong answer" for a broken chain.
            "num_hops": num_hops,
            "hop_depths": depths,
            "terminal_depth": terminal_depth,
            "tail_anchored_hops": max(num_hops - len(depths), 0),
            "terminal_tail_anchored": terminal_depth is None,
            "terminal_value": final_value,
        },
    )
