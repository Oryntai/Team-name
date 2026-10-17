"""Independent label/prediction evaluator. Intentionally imports no decision engine."""

import argparse
import json
import re
from dataclasses import dataclass
from .contracts import number, text


def validate_splits(records):
    if not isinstance(records, list):
        raise ValueError("split manifest must be an array")
    seen = {}
    video_ids = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("manifest row must be an object")
        video_id = text(row.get("video_id"), "video_id")
        if video_id in video_ids:
            raise ValueError(f"duplicate video_id: {video_id}")
        video_ids.add(video_id)
        if row.get("split") not in ("train", "dev", "test"):
            raise ValueError("invalid split")
        digest = text(row.get("sha256"), "sha256")
        if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        group = text(row.get("group_id"), "group_id")
        for field, value in (("sha256", digest.lower()), ("group_id", group)):
            key = (field, value)
            if key in seen and seen[key] != row["split"]:
                raise ValueError(f"{field} crosses splits: {value}")
            seen[key] = row["split"]
    return True


def match_events(predicted, labels, tolerance):
    """Maximum-cardinality, minimum-total-time-error bipartite assignment.

    Successive shortest augmenting paths in a residual network; predictions are
    never reused. Type/rule labels are assessed AFTER temporal matching.
    """
    number(tolerance, "tolerance")
    if tolerance < 0:
        raise ValueError("negative tolerance")
    prediction_count = len(predicted)
    label_count = len(labels)
    for event in [*predicted, *labels]:
        if number(event["timecode_sec"], "timecode_sec") < 0:
            raise ValueError("negative event time")
    source = prediction_count + label_count
    sink = source + 1
    node_count = sink + 1

    @dataclass
    class ResidualEdge:
        target: int
        reverse_index: int
        capacity: int
        cost: float

    graph: list[list[ResidualEdge]] = [[] for _ in range(node_count)]

    def add_edge(origin, target, cost):
        forward = ResidualEdge(target, len(graph[target]), 1, cost)
        reverse = ResidualEdge(origin, len(graph[origin]), 0, -cost)
        graph[origin].append(forward)
        graph[target].append(reverse)

    for prediction_index in range(prediction_count):
        add_edge(source, prediction_index, 0)
    for label_index in range(label_count):
        add_edge(prediction_count + label_index, sink, 0)
    for prediction_index, prediction in enumerate(predicted):
        for label_index, label in enumerate(labels):
            error = abs(prediction["timecode_sec"] - label["timecode_sec"])
            if error <= tolerance:
                add_edge(prediction_index, prediction_count + label_index, error)
    while True:
        distances = [float("inf")] * node_count
        previous = [None] * node_count
        distances[source] = 0
        # Bellman-Ford handles negative reverse costs after augmentation.
        for _ in range(node_count - 1):
            changed = False
            for origin in range(node_count):
                for edge_index, edge in enumerate(graph[origin]):
                    candidate_distance = distances[origin] + edge.cost
                    if edge.capacity and candidate_distance < distances[edge.target]:
                        distances[edge.target] = candidate_distance
                        previous[edge.target] = (origin, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            origin, edge_index = previous[node]
            edge = graph[origin][edge_index]
            edge.capacity = 0
            graph[node][edge.reverse_index].capacity = 1
            node = origin
    matches = []
    for prediction_index in range(prediction_count):
        for edge in graph[prediction_index]:
            is_label_edge = prediction_count <= edge.target < source
            if is_label_edge and edge.capacity == 0:
                label_index = edge.target - prediction_count
                error = abs(
                    predicted[prediction_index]["timecode_sec"]
                    - labels[label_index]["timecode_sec"]
                )
                matches.append((prediction_index, label_index, error))
    return matches


def ratios(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def evaluate(predictions, annotations):
    """Evaluate temporal events and binary outcomes with explicit label scopes."""

    def index(rows):
        if not isinstance(rows, list):
            raise ValueError("records must be an array")
        result = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("record must be an object")
            key = text(row.get("video_id"), "video_id")
            if key in result:
                raise ValueError("duplicate video_id")
            result[key] = row
        return result

    preds, labels = index(predictions), index(annotations)
    for row in labels.values():
        if "exhaustive" in row and type(row["exhaustive"]) is not bool:
            raise ValueError("exhaustive must be bool")
        if row.get("verdict") not in (None, "violation", "no_violation"):
            raise ValueError("invalid label verdict")
        if row.get("verdict") == "no_violation" and row.get("events"):
            raise ValueError("negative ground truth contradicts positive events")
    for row in [*preds.values(), *labels.values()]:
        if "events" in row:
            if not isinstance(row["events"], list):
                raise ValueError("events must be an array")
            for event in row["events"]:
                if not isinstance(event, dict):
                    raise ValueError("event must be an object")
                if number(event.get("timecode_sec"), "timecode_sec") < 0:
                    raise ValueError("negative event time")
    for row in preds.values():
        if row.get("status") == "complete" and row.get("verdict") in (
            "violation",
            "no_violation",
        ):
            if "events" not in row:
                raise ValueError("complete prediction requires events")
            if (row["verdict"] == "violation") != bool(row["events"]):
                raise ValueError("verdict/event inconsistency")

    counts = dict(
        TP=0,
        FP=0,
        FN=0,
        TN=0,
        failures=0,
        missing_predictions=0,
        partial_predictions=0,
        failed_predictions=0,
        invalid_predictions=0,
        completed=0,
        total=len(preds.keys() | labels.keys()),
        missing_labels=0,
    )
    event_stats = {}
    for tolerance in (1, 3, 5):
        event_stats[str(tolerance)] = {
            "all_known_labels": {
                "matched": 0,
                "unmatched_labels": 0,
                "ignored_extra_predictions": 0,
                "time_errors_sec": [],
                "type_correct": 0,
                "type_total": 0,
                "rule_correct": 0,
                "rule_total": 0,
            },
            "exhaustive_labels": {"TP": 0, "FP": 0, "FN": 0},
        }
    for vid in sorted(preds.keys() | labels.keys()):
        prediction, label = preds.get(vid), labels.get(vid)
        failed = (
            prediction is None
            or prediction.get("status") != "complete"
            or prediction.get("verdict") not in ("violation", "no_violation")
        )
        if failed:
            counts["failures"] += 1
            if prediction is None:
                counts["missing_predictions"] += 1
            elif prediction.get("status") == "partial":
                counts["partial_predictions"] += 1
            elif prediction.get("status") in ("failed", "incomplete"):
                counts["failed_predictions"] += 1
            else:
                counts["invalid_predictions"] += 1
        else:
            counts["completed"] += 1
        if label is None or label.get("verdict") is None:
            counts["missing_labels"] += 1
        elif not failed:
            positive = prediction["verdict"] == "violation"
            truth = label["verdict"] == "violation"
            if positive and truth:
                outcome = "TP"
            elif positive:
                outcome = "FP"
            elif truth:
                outcome = "FN"
            else:
                outcome = "TN"
            counts[outcome] += 1
        # A binary positive may have no event annotations at all.
        if label is None or "events" not in label:
            continue
        predicted_events = [] if failed else prediction["events"]
        labelled_events = label["events"]
        for tolerance in (1, 3, 5):
            scopes = event_stats[str(tolerance)]
            known = scopes["all_known_labels"]
            pairs = match_events(predicted_events, labelled_events, tolerance)
            unmatched_labels = len(labelled_events) - len(pairs)
            unmatched_predictions = len(predicted_events) - len(pairs)
            known["matched"] += len(pairs)
            known["unmatched_labels"] += unmatched_labels
            if label.get("exhaustive", False):
                exhaustive = scopes["exhaustive_labels"]
                exhaustive["TP"] += len(pairs)
                exhaustive["FP"] += unmatched_predictions
                exhaustive["FN"] += unmatched_labels
            else:
                known["ignored_extra_predictions"] += unmatched_predictions
            for i, j, error in pairs:
                known["time_errors_sec"].append(error)
                for field, prefix in (("type", "type"), ("rule_id", "rule")):
                    if labelled_events[j].get(field) is not None:
                        known[prefix + "_total"] += 1
                        known[prefix + "_correct"] += (
                            predicted_events[i].get(field) == labelled_events[j][field]
                        )
    for scopes in event_stats.values():
        known = scopes["all_known_labels"]
        total_labels = known["matched"] + known["unmatched_labels"]
        known["recall_of_known_labels"] = (
            known["matched"] / total_labels if total_labels else None
        )
        errors = known["time_errors_sec"]
        known["mean_time_error_sec"] = sum(errors) / len(errors) if errors else None
        for prefix in ("type", "rule"):
            known[prefix + "_accuracy"] = (
                known[prefix + "_correct"] / known[prefix + "_total"]
                if known[prefix + "_total"]
                else None
            )
        exhaustive = scopes["exhaustive_labels"]
        exhaustive.update(ratios(exhaustive["TP"], exhaustive["FP"], exhaustive["FN"]))
    counts["completion_rate"] = (
        counts["completed"] / counts["total"] if counts["total"] else None
    )
    return {
        "video": {**counts, **ratios(counts["TP"], counts["FP"], counts["FN"])},
        "events": event_stats,
        "scopes": {
            "video": "Confusion matrix and precision/recall/F1 conditional on complete predictions with supplied binary gold labels; event exhaustiveness does not affect binary gold validity. Completion/failure counts cover union of prediction and label video IDs.",
            "events.exhaustive_labels": "Precision/recall/F1 and TP/FP/FN use only videos with explicit exhaustive event annotations. Failed analyses retain FN labels.",
            "events.all_known_labels": "Known-label recall, timing and classification cover all supplied event annotations. Extras on non-exhaustive videos are ignored, never FP. No precision/F1 is estimated for this scope.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("predictions")
    ev.add_argument("annotations")
    split = sub.add_parser("validate-splits")
    split.add_argument("manifest")
    args = parser.parse_args()

    def read(path):
        with open(path) as f:
            return json.load(f)

    output = (
        evaluate(read(args.predictions), read(args.annotations))
        if args.command == "evaluate"
        else {"valid": validate_splits(read(args.manifest))}
    )
    print(json.dumps(output, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
