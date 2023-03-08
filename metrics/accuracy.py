import torch


class Accuracy:
    def __init__(self):
        self._matches = None
        self._totals = None

    def evaluate(self, predictions, masks):
        with torch.no_grad():
            # create one-hot encoding of masks for each class
            masks_one_hot = torch.zeros_like(predictions)
            masks_one_hot.scatter_(1, masks.unsqueeze(dim=1), 1)

            # create one-hot encoding of predicted labels for each class
            predicted_one_hot = torch.zeros_like(predictions)
            predicted_one_hot.scatter_(1, torch.argmax(predictions, dim=1, keepdim=True), 1)
            
            # compute matches and totals for all classes
            matches = (predicted_one_hot * masks_one_hot).sum(dim=(2, 3))
            totals = masks_one_hot.sum(dim=(2, 3))

            if self._matches is None:
                self._matches = matches.sum(dim=0)
            else:
                self._matches += matches.sum(dim=0)

            if self._totals is None:
                self._totals = totals.sum(dim=0)
            else:
                self._totals += totals.sum(dim=0)

    def accuracy(self):
        if self._matches is None or self._totals is None:
            return None
        return self._matches / (self._totals + 1e-6) # adding epsilon to avoid divide by zero errors

    def mean_accuracy(self):
        if self._matches is None or self._totals is None:
            return None
        return self._matches.sum(dim=0) / (self._totals.sum(dim=0) + 1e-6) # adding epsilon to avoid divide by zero errors
