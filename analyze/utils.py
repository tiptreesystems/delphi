from collections import defaultdict

import numpy as np


def analyze_forecast_results(sampled_questions, loaded_fcasts_with_examples, loaded_fcasts_no_examples,
                            loader, selected_resolution_date, forecast_due_date):
    """Analyze forecast results and compute aggregate statistics."""
    rng = np.random.default_rng(42)
    sf_aggregate = defaultdict(list)
    q_aggregate = defaultdict(list)
    qid_to_label = {}

    def _extract_forecasts_from_payload(payload):
        """Extract forecasts from various payload formats."""
        if isinstance(payload, dict):
            if "forecast" in payload and isinstance(payload["forecast"], dict) and "forecasts" in payload["forecast"]:
                return payload["forecast"]["forecasts"]
            if "forecasts" in payload and isinstance(payload["forecasts"], (list, tuple)):
                return payload["forecasts"]
        if isinstance(payload, (list, tuple)):
            return payload
        raise ValueError(f"Unrecognized payload shape for forecasts: {type(payload)}")

    def _safe_super_gt(qid_, sfid_):
        """Safely get ground truth for a superforecaster."""
        try:
            gt = loader.get_super_forecasts(
                question_id=qid_,
                user_id=sfid_,
                resolution_date=selected_resolution_date
            )
            if gt:
                val = getattr(gt[0], "forecast", gt[0])
                return getattr(val, "value", val)
        except Exception as e:
            print(f"[warn] ground truth missing for qid={qid_}, sfid={sfid_}: {e}")
        return None

    for q in sampled_questions:
        qid = q.id

        sf_payloads = loaded_fcasts_with_examples.get(qid, [])
        base_payloads = loaded_fcasts_no_examples.get(qid, [])

        if not sf_payloads or not base_payloads:
            print(f"[skip] Missing data for qid={qid}")
            continue

        # Process baseline forecasts once
        base_item = base_payloads[0]
        try:
            base_forecasts = _extract_forecasts_from_payload(base_item)
        except Exception as e:
            print(f"[skip] qid={qid} baseline extract error: {e}")
            continue

        # Collect data for each superforecaster
        for payload in sf_payloads:
            sfid = payload.get("subject_id", "unknown_sfid")
            gt_val = _safe_super_gt(qid, sfid)
            if gt_val is None:
                continue

            try:
                sf_forecasts = _extract_forecasts_from_payload(payload)
            except Exception:
                continue
            if not sf_forecasts:
                continue

            # Calculate absolute errors
            ae_sf = np.abs(np.asarray(sf_forecasts, dtype=float) - float(gt_val))
            ae_base = np.abs(np.asarray(base_forecasts, dtype=float) - float(gt_val))

            if len(ae_sf) and len(ae_base):
                sf_aggregate[sfid].append((ae_sf, ae_base))
                q_aggregate[qid].append((ae_sf, ae_base))

        # Store question label
        if qid not in qid_to_label:
            pretty = q.question.replace("{resolution_date}", selected_resolution_date).replace("{forecast_due_date}", forecast_due_date)
            qid_to_label[qid] = pretty[:80] + ("…" if len(pretty) > 80 else "")

    return sf_aggregate, q_aggregate, qid_to_label
