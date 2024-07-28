import torch
from collections import Counter


class Inferencer:
    def __init__(self, model=None, test_loader=None, device=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device


    def infer(self):
        self.model.eval()

        truths = []
        preds = []
        hit_accumulated = 0
        total_items = 0

        difference_counter = Counter()

        with torch.no_grad():
            for idx, (features_b, labels_b, mask_b) in enumerate(self.test_loader):
                features_b = features_b.to(self.device)
                labels_b = labels_b.to(self.device)
                mask_b = mask_b.to(self.device)

                output_b = self.model(features_b)

                output_flat = output_b.reshape(-1)
                labels_flat = labels_b.reshape(-1)
                mask_flat = mask_b.reshape(-1)

                output_masked = output_flat[mask_flat == 1]
                labels_masked = labels_flat[mask_flat == 1]

                pred = torch.round(output_masked)
                hit_count = (pred == labels_masked).sum().item()
                hit_accumulated += hit_count
                total_items += len(labels_masked)

                difference = torch.abs(labels_masked - pred)
                for d in difference:
                    difference_counter[d.item()] += 1

                truths.append(labels_masked)
                preds.append(pred)
            
        return torch.cat(preds), torch.cat(truths), hit_accumulated/total_items, difference_counter
    
