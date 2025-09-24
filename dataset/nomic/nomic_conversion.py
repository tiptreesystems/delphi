import json
import jsonlines

with open("2024-07-21-human.json", "r") as f:
    data = json.load(f)

with jsonlines.open("nomic_upload.jsonl", "w") as writer:
    for q in data["questions"]:
        item = {
            "id": q["id"],
            "text": f"{q['question']} {q['background']}".strip(),
            "source": q.get("source", ""),
            "url": q.get("url", ""),
            "probability": q.get("freeze_datetime_value", None),
        }
        writer.write(item)
