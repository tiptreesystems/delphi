import os
import pickle
import numpy as np
import re

import debugpy

print("Starting debugpy...")
debugpy.listen(5679)  # Adjust port as needed
debugpy.wait_for_client()  # Wait for the debugger to attach


forecast_due_date = "2024-07-21"  # Example date, adjust as needed
selected_resolution_date = "2025-07-21"
initial_forecasts_path = "outputs_initial_forecasts_flexible_retry"

if __name__ == "__main__":
    pkl_files = [
        f
        for f in os.listdir(f"{initial_forecasts_path}/")
        if f.startswith("collected_fcasts")
        and f.endswith(".pkl")
        and f"{selected_resolution_date}" in f
    ]

    # Split pkl files into with_examples and no_examples
    with_examples_files = [
        f
        for f in pkl_files
        if f.startswith("collected_fcasts_with_examples")
        and f"{selected_resolution_date}" in f
    ]
    no_examples_files = [
        f
        for f in pkl_files
        if f.startswith("collected_fcasts_no_examples")
        and f"{selected_resolution_date}" in f
    ]

    loaded_fcasts_with_examples = {}
    for fname in with_examples_files:
        # Extract question id between 'collected_fcasts_' and '.pkl'
        qid = fname[
            len(f"collected_fcasts_with_examples_{selected_resolution_date}_") : -len(
                ".pkl"
            )
        ]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_fcasts_with_examples[qid] = [q for q in pickle.load(f)]

    loaded_fcasts_no_examples = {}
    for fname in no_examples_files:
        # Extract question id between 'collected_fcasts_no_examples_' and '.pkl'
        qid = fname[
            len(f"collected_fcasts_no_examples_{selected_resolution_date}_") : -len(
                ".pkl"
            )
        ]
        with open(f"{initial_forecasts_path}/{fname}", "rb") as f:
            loaded_fcasts_no_examples[qid] = [q for q in pickle.load(f)]

    # Estimate proportion of forecasts equal to exactly 0.5 in both loaded_fcasts dicts
    def _extract_forecasts_generic(payload):
        """
        Extract a list of numeric forecasts from various payload shapes:
          - {'forecast': {'forecasts': [...]}}
          - {'forecasts': [...]}
          - [...] (already a list/tuple of floats)
        """
        try:
            if isinstance(payload, dict):
                if (
                    "forecast" in payload
                    and isinstance(payload["forecast"], dict)
                    and "forecasts" in payload["forecast"]
                ):
                    return payload["forecast"]["forecasts"]
                if "forecasts" in payload and isinstance(
                    payload["forecasts"], (list, tuple)
                ):
                    return payload["forecasts"]
            if isinstance(payload, (list, tuple)):
                return payload
        except Exception:
            pass
        return []

    def _count_exact_half(fcasts_dict):
        n_half = 0
        n_total = 0
        for _, items in fcasts_dict.items():
            for payload in items:
                forecasts = _extract_forecasts_generic(payload)
                for v in forecasts:
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    if np.isnan(val):
                        continue
                    n_total += 1
                    if np.isclose(val, 0.5, atol=1e-3):
                        n_half += 1
        prop = (n_half / n_total) if n_total else float("nan")
        return n_half, n_total, prop

    half_w_n, w_total_forecasts, w_prop_forecasts = _count_exact_half(
        loaded_fcasts_with_examples
    )
    half_b_n, b_total_forecasts, b_prop_forecasts = _count_exact_half(
        loaded_fcasts_no_examples
    )

    print(
        f"With-examples (forecasts): {half_w_n}/{w_total_forecasts} = {w_prop_forecasts:.2%} are exactly 0.5"
    )
    print(
        f"No-examples  (forecasts): {half_b_n}/{b_total_forecasts} = {b_prop_forecasts:.2%} are exactly 0.5"
    )

    both_half = half_w_n + half_b_n
    both_total = w_total_forecasts + b_total_forecasts
    both_prop = (both_half / both_total) if both_total else float("nan")
    print(
        f"Combined     (forecasts): {both_half}/{both_total} = {both_prop:.2%} are exactly 0.5"
    )

    # Now check "FINAL PROBABILITY" at the per-message level:
    # - For each conversation, skip the first message (index 0).
    # - Among the remaining messages, only consider assistant messages.
    # - Count how many assistant messages contain a final probability.
    # pattern = re.compile(r'final\s*p    robability\s*:\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
    pattern = re.compile(r"FINAL PROBABILITY:\s*(0?\.\d+|1\.0|0|1)", re.IGNORECASE)

    def _normalize_msg_text(msg):
        role = None
        text = ""
        if isinstance(msg, dict):
            role = (
                (msg.get("role") or "").lower()
                if isinstance(msg.get("role"), str)
                else None
            )
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, str):
                        parts.append(c)
                    elif isinstance(c, dict):
                        if isinstance(c.get("text"), str):
                            parts.append(c["text"])
                        elif isinstance(c.get("content"), str):
                            parts.append(c["content"])
                text = " ".join(parts)
            elif isinstance(msg.get("text"), str):  # some libs store text directly
                text = msg["text"]
        elif isinstance(msg, str):
            # Role unknown; we'll skip role filtering for strings
            role = None
            text = msg
        return role, text

    # def _extract_all_conversations(payload):
    #     """
    #     Return a list of conversations (each conversation = list of messages).
    #     Supports both:
    #     - New format: full_conversation = [conv1, conv2, ...]
    #     - Legacy:     full_conversation = conv  (single list) -> wrapped as [conv]
    #     """
    #     try:
    #         if isinstance(payload, dict):
    #             fc = payload.get("forecast")
    #             if isinstance(fc, dict) and "full_conversation" in fc:
    #                 convs = fc["full_conversation"]
    #             elif "full_conversation" in payload:
    #                 convs = payload["full_conversation"]
    #             else:
    #                 return []

    #             if not isinstance(convs, list):
    #                 return []

    #             # If it's a single conversation (list of messages), wrap it.
    #             if len(convs) == 0:
    #                 return []
    #             if all(isinstance(m, (dict, str)) for m in convs):
    #                 return [convs]
    #             # Otherwise expect a list of conversations (each a list)
    #             if all(isinstance(c, list) for c in convs):
    #                 return convs

    #             return []
    #     except Exception:
    #         pass
    #     return []

    def _iter_assistant_msgs_after_first(conv):
        if not isinstance(conv, list):
            return
        for i, msg in enumerate(conv):
            if i == 0:
                continue
            role, text = _normalize_msg_text(msg)
            # Only count assistant messages; if role unknown, skip to keep counts conservative
            if role == "assistant":
                yield text

    def _count_final_probability_messages(fcasts_dict):
        """
        Strict: relies on new format and validated structures.
        Counts assistant messages (post-first) matching `pattern`.
        """
        if not isinstance(fcasts_dict, dict):
            raise TypeError(f"fcasts_dict must be dict, got {type(fcasts_dict)}")

        hits = 0
        total_msgs = 0

        for qid, items in fcasts_dict.items():
            if not isinstance(items, list):
                raise TypeError(
                    f"Value for key {qid!r} must be list of payloads, got {type(items)}"
                )
            for payload in items:
                convs = payload["full_conversation"]
                for conv in convs:
                    for text in _iter_assistant_msgs_after_first(conv):
                        total_msgs += 1
                        if pattern.search(text):
                            hits += 1
                        else:
                            # print(f"\n--- full_conversation for qid={qid} ---\n{convs}\n")
                            continue

        prop = (hits / total_msgs) if total_msgs else float("nan")
        return hits, total_msgs, prop

    w_hits_msgs, w_total_msgs, w_prop_msgs = _count_final_probability_messages(
        loaded_fcasts_with_examples
    )
    b_hits_msgs, b_total_msgs, b_prop_msgs = _count_final_probability_messages(
        loaded_fcasts_no_examples
    )

    print(
        f"FINAL PROBABILITY (assistant messages) — With-examples: {w_hits_msgs}/{w_total_msgs} = {w_prop_msgs:.2%}"
    )
    print(
        f"FINAL PROBABILITY (assistant messages) — No-examples:  {b_hits_msgs}/{b_total_msgs} = {b_prop_msgs:.2%}"
    )

    combined_hits_msgs = w_hits_msgs + b_hits_msgs
    combined_total_msgs = w_total_msgs + b_total_msgs
    combined_prop_msgs = (
        (combined_hits_msgs / combined_total_msgs)
        if combined_total_msgs
        else float("nan")
    )
    print(
        f"FINAL PROBABILITY (assistant messages) — Combined:     {combined_hits_msgs}/{combined_total_msgs} = {combined_prop_msgs:.2%}"
    )
