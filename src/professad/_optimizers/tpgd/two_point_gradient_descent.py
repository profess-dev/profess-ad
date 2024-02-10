import torch
from torch.optim.optimizer import Optimizer

class TPGD(Optimizer):
  """
  Two-point gradient descent algorithm 
  [IMA Journal of Numerical Analysis, Volume 8, Issue 1, January 1988, Pages 141–148 <https://doi.org/10.1093/imanum/8.1.141>]
  """

  def __init__(self, params, lr=1e-1):
    """
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining parameter groups
      lr (float)       : initial learning rate / gradient descent step-size
    """
    if lr <= 0.0: raise ValueError("Invalid initial learning rate: {} - should be > 0".format(lr))

    defaults = dict(lr=lr)
    super(TPGD, self).__init__(params, defaults)
    assert len(self.param_groups) == 1, ValueError("TPGD doesn't support per-parameter options (parameter groups)")

    self.iter = 0
    self._params = self.param_groups[0]['params']
    
  def step(self, closure=None):
    """
    Performs a single optimization step.

    Args:
        closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()
   
    numerator = 0; denominator = 0  
  
    for p in self._params:
      
      if p.grad is None:
        continue
  
      state = self.state[p]
         
      if self.iter != 0: 
        dx = p.data - state['x_prev']
        dg = p.grad.data - state['g_prev']
        numerator += torch.sum(dx * dx).item()
        denominator += torch.sum(dx * dg).item()
      
      state['x_prev'] = p.data.clone(memory_format=torch.preserve_format)
      state['g_prev'] = p.grad.data.clone(memory_format=torch.preserve_format)
            
    if self.iter == 0 or denominator == 0:  
      alpha = self.param_groups[0]['lr']
    else: 
      alpha = numerator / denominator
      if alpha <= 0: alpha = self.param_groups[0]['lr'] # avoid finding maximas
      
    for p in self._params:
      p.data.add_(p.grad.data, alpha = - alpha)
    
    self.iter += 1
    
    return loss
