"""Training-label priors that cannot access image pixels."""
from collections import defaultdict
import torch
from .data import rebin, render


class QuestionPriors:
    def __init__(self, rows):
        self.buckets = defaultdict(lambda: {"n": 0, "sum": 0.0, "hist": torch.zeros(5),
                                           "wins": defaultdict(float), "offered": defaultdict(int)})
        for row in rows:
            for key in self.keys(row):
                part = self.buckets[key]; part["n"] += 1
                if row["type"] == "noul": part["sum"] += row["value"]
                elif row["type"] == "score": part["hist"] += rebin(row["hist"], 5)
                else:
                    criteria = row.get("criteria", row.get("candidates"))
                    names = list(criteria)
                    label = row.get("label", 0)
                    if isinstance(label, int): label = names[label]
                    for name in names:
                        desc = render(criteria[name]) if isinstance(criteria, dict) else name
                        part["offered"][desc] += 1
                        part["wins"][desc] += row.get("target_probs", {}).get(name, float(name == label))

    @staticmethod
    def keys(row):
        q = render(row["instructions"]).lower()
        return [(row.get("source", "unknown"), row["type"], q),
                (row.get("source", "unknown"), row["type"], "family:" + " ".join(q.split()[:2])),
                (row.get("source", "unknown"), row["type"], "*")]

    def predict(self, row, ex):
        part = next((self.buckets[k] for k in self.keys(row) if k in self.buckets and self.buckets[k]["n"] >= 3), None)
        if part is None: return torch.full_like(ex.target, 1 / len(ex.target))
        if row["type"] == "noul":
            p = (part["sum"] + 1) / (part["n"] + 2)
            return torch.tensor([1 - p, p])
        if row["type"] == "score":
            return rebin(part["hist"], len(ex.target))
        criteria = row.get("criteria", row.get("candidates"))
        values = []
        for name in ex.names:
            desc = render(criteria[name]) if isinstance(criteria, dict) else name
            values.append((part["wins"].get(desc, 0) + 1) / (part["offered"].get(desc, 0) + 2))
        p = torch.tensor(values); return p / p.sum()
