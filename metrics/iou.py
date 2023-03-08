import torch


class IoU:
    def __init__(self):
        self._intersection = None
        self._union = None

    def evaluate(self, predictions, masks):
        with torch.no_grad():
            # create one-hot encoding of masks for each class
            masks_one_hot = torch.zeros_like(predictions)
            masks_one_hot.scatter_(1, masks.unsqueeze(dim=1), 1)

            # create one-hot encoding of predicted labels for each class
            predicted_one_hot = torch.zeros_like(predictions)
            predicted_one_hot.scatter_(1, torch.argmax(predictions, dim=1, keepdim=True), 1)
            
            # compute intersection and union for all classes
            intersection = (predicted_one_hot * masks_one_hot).sum(dim=(2, 3))
            union = predicted_one_hot.sum(dim=(2, 3)) + masks_one_hot.sum(dim=(2, 3)) - intersection

            if self._intersection is None:
                self._intersection = intersection.sum(dim=0)
            else:
                self._intersection += intersection.sum(dim=0)

            if self._union is None:
                self._union = union.sum(dim=0)
            else:
                self._union += union.sum(dim=0)

    def iou(self):
        if self._intersection is None or self._union is None:
            return None
        return self._intersection / (self._union + 1e-6) # adding epsilon to avoid divide by zero errors

    def mean_iou(self):
        if self._intersection is None or self._union is None:
            return None
        return self._intersection.sum(dim=0) / (self._union.sum(dim=0) + 1e-6) # adding epsilon to avoid divide by zero errors
