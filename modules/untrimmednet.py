import torch
from torch import nn
import torch.nn.functional as F
from .two_stream import TwoStreamNet

def soft_selection(class_scores, selection_scores):
    """
    This function performs soft selection of instances.

    Args:
        class_scores (torch.Tensor): The classification scores for the instances,
                                      shape should be [num_instances, num_classes].
        selection_scores (torch.Tensor): The selection scores (from attention mechanism) for the instances,
                                          shape should be [num_instances, 1].

    Returns:
        torch.Tensor: The aggregated score for each class, shape [num_classes].
    """
    # Ensure selection_scores is properly sized
    selection_scores = selection_scores.squeeze()

    # Apply softmax to the selection scores
    normalized_weights = F.softmax(selection_scores, dim=0)

    # Multiply the normalized weights with the classification scores
    weighted_scores = class_scores * normalized_weights.unsqueeze(1)  # Add an axis for proper broadcasting

    # Sum the scores for each class
    aggregated_scores = torch.sum(weighted_scores, dim=0)  # Sum over instances

    return aggregated_scores

def hard_selection(scores, k):
    """
    This function performs hard selection of instances.

    Args:
        scores (torch.Tensor): The classification scores for the instances,
                               shape should be [num_classes, num_instances].
        k (int): The number of top instances to select for each class.

    Returns:
        torch.Tensor: A binary mask indicating the selected instances,
                      shape [num_classes, num_instances].
    """
    # Check if k is greater than the number of instances
    num_instances = scores.shape[1]
    if k > num_instances:
        raise ValueError(f"k should be less than or equal to the number of instances, but got k={k} and num_instances={num_instances}")

    # Sort scores and get the indices of the top k scores
    _, indices = torch.topk(scores, k, dim=1)

    # Create a binary selection mask
    selection_mask = torch.zeros_like(scores, dtype=torch.bool)

    # Mark the top k instances as True for each class
    for i in range(scores.shape[0]):  # Loop through each class
        selection_mask[i, indices[i]] = True

    return selection_mask

class UntrimmedNets(nn.Module):
  def __init__(self, num_classes, selection_module = "hard_selection")->None:
    super().__init__()
    self.feature_extraction = TwoStreamNet(num_classes, is_extractor=True)
    self.classification = nn.Linear(2048, num_classes)
    assert selection_module in ["hard_selection", "soft_selection"]
    if selection_module == "soft_selection":
      self.attention = nn.Linear(2048, 1)
    self.selection_module = selection_module

  def forward(self, X: tuple[torch.Tensor, torch.tensor]) -> torch.Tensor:
    embedding_vector = self.feature_extraction(X)
    class_scores = self.classification(embedding_vector)
    if self.selection_module == "soft_selection":
      selection_scores = self.attention(embedding_vector)
      aggregated_scores = soft_selection(class_scores, selection_scores)
      return aggregated_scores
    return class_scores