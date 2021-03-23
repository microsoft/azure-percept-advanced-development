import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

import numpy as np


def dice(im1, im2):
  """
  Computes the Dice coefficient, a measure of set similarity.
  Parameters
  ----------
  im1 : array-like, bool
      Any array of arbitrary size. If not boolean, will be converted.
  im2 : array-like, bool
      Any other array of identical size. If not boolean, will be converted.
  Returns
  -------
  dice : float
      Dice coefficient as a float on range [0,1].
      Maximum similarity = 1
      No similarity = 0
      
  Notes
  -----
  The order of inputs for `dice` is irrelevant. The result will be
  identical if `im1` and `im2` are switched.
  """
  im1 = np.asarray(im1).astype(np.bool)
  im2 = np.asarray(im2).astype(np.bool)

  if im1.shape != im2.shape:
      raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

  # Compute Dice coefficient
  intersection = np.logical_and(im1, im2)

  return 2. * intersection.sum() / (im1.sum() + im2.sum())
