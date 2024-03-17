Fast Fourier Transform (FFT) Tools
==================================

The following are utility functions for the implementation of various functionals via
fast Fourier transform (FFT) related operations. For example, the gradient terms used in semi-local
kinetic functionals, such as the :ref:`Luo-Karasiev-Trickey (LKT) functional <lkt>`, are implemented using FFTs. 
The following code block shows how these FFT tools can be used to implement the LKT functional, given by

.. math:: T_\text{LKT}[n] = T_\text{vW} + \int_\Omega d^3\mathbf{r} \frac{\tau_\text{TF}(\mathbf{r})}{\cosh(1.3 s)}

where :math:`T_\text{vW}` is the :ref:`von Weizsaecker functional <vw>`, :math:`\tau_\text{TF}(\mathbf{r})` is 
the :ref:`Thomas-Fermi <tf>` kinetic energy density and :math:`s` is the :ref:`reduced gradient <s>`. ::

  def LuoKarasievTrickey(box_vecs, den): 
      TF_ked = 0.3 * (3 * np.pi * np.pi)**(2 / 3) * den.pow(5 / 3)
      kxyz = wavevectors(box_vecs, den.shape)
      s = reduced_gradient(kxyz, den)
      # clamp to avoid s from growing too large, which can cause F_pauli -> 0 and get nan derivatives
      F_pauli = 1 / torch.cosh(1.3 * s.clamp(max=100))
      pauli_T = torch.mean(TF_ked * F_pauli) * torch.abs(torch.linalg.det(box_vecs))
      return Weizsaecker(box_vecs, den) + pauli_T


FFTs are also useful for functionals that involve real space convolutions. Besides the non-local kinetic functionals,
the more widely known :ref:`Hartree functional <hart>` given by

.. math:: U_\text{Hartree}[n] = \frac{1}{2} \int d^3\mathbf{r} d^3\mathbf{r}'~ 
          \frac{n(\mathbf{r})n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} 

also involves convolutions, and hence benefits from an FFT-based implementation motivated by the convolution theorem. ::

  def Hartree(box_vecs, den):
      k2 = wavevectors(box_vecs, den.shape).square().sum(-1)
      den_ft = torch.fft.rfftn(den)
      coloumb_ft = torch.zeros(k2.shape, dtype=torch.double, device=den.device)
      # set k=0 component to zero. appropriate if the density integrates to
      # zero over the box (e.g. if neutralized by a uniform background charge).
      coloumb_ft[k2 != 0] = 4 * np.pi / k2[k2 != 0]
      pot = torch.fft.irfftn(den_ft * coloumb_ft, den.shape)
      return 0.5 * torch.mean(den * pot) * torch.abs(torch.linalg.det(box_vecs))


Wavevector Tools
----------------

Wavevectors
^^^^^^^^^^^

.. autofunction:: functional_tools.wavevectors


Gradient Squared
^^^^^^^^^^^^^^^^

.. autofunction:: functional_tools.grad_dot_grad

Laplacian
^^^^^^^^^

.. autofunction:: functional_tools.laplacian


Density Descriptors
-------------------

Reduced Gradient
^^^^^^^^^^^^^^^^

.. _s:

.. autofunction:: functional_tools.reduced_gradient

Reduced Gradient Squared
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: functional_tools.reduced_gradient_squared

Reduced Laplacian
^^^^^^^^^^^^^^^^^

.. _q:

.. autofunction:: functional_tools.reduced_laplacian

