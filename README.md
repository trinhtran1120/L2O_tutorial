# 2026-L4DC-Tutorial-Learning-to-Optimize
2026 L4DC tutorial on learning to optimize, as part of the tutorial session "Scientific Machine Learning for Modeling, Optimization, and Control".


## Resources
- Good tutorial session as an example: https://implicit-layers-tutorial.org/
- L2O examples from Neuromancer: https://github.com/pnnl/neuromancer
  - General description: https://github.com/pnnl/neuromancer/tree/develop/examples/parametric_programming
  - Simple L2O example using differentiable parametric programming (DPP) = self-supervised learning with soft-constraint loss: https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_1_basics.ipynb
  - Similar simple examples for QP and QCQP with soft-constraint loss (probably not very useful for my tutorial): https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_2_pQP.ipynb
  - Similar examples for NLPs with different objective functions with soft-constraint loss (probably not very useful for my tutorial): https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_3_pNLP.ipynb
  - Same self-supervised method but using unrolled projected gradient for improving predicted solution and constraint satisfaction (can mention as a method, but not as good as ours): https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_4_projectedGradient.ipynb
  - Same self-supervised method but using cvxpylayer to create a differentiable solution projection/correction layer onto the constraint set (may present): https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_5_cvxpy_layers.ipynb
  - Use self-supervised learning L2O to predict initial solution, then apply an operator splitting method (like DR or ADMM) to correct and find the final optimal solution, but the proximal operator is trainable (the l2 norm becomes M-norm where optimal M > 0 is predicted from parameters); however it doesn't learn the proximal operator as in our method (could be interesting to at least mention): https://colab.research.google.com/github/pnnl/neuromancer/blob/master/examples/parametric_programming/Part_6_pQp_lopoCorrection.ipynb
- Slides: perhaps most useful for this L2O tutorial are those marked with (*).
  - (*) Jan's course on SciML, lecture on L2O: https://github.com/drgona/SciML-Course/tree/main/week_5
  - (*) Jan's course on SciML, lecture on feasibility layer (including L2O for MIP): https://github.com/drgona/SciML-Course/tree/main/week_10
  - Jan's course on SciML, lecture on differentiable optimization: https://github.com/drgona/SciML-Course/tree/main/week_9
  - Jan's slides on SciML (L2M, L2O, L2C) have some slides on L2O with constraints, but pretty high level: https://github.com/drgona/slides/blob/main/2025/Drgona_SciML_INFORMS.pptx. Other slide decks have more contents (like L2O MIP, PINN, PDE), such as https://github.com/drgona/slides/blob/main/2026/ICCPS/Drgona_SciML_ICCPS_workshop.pptx and https://github.com/drgona/slides/blob/main/2026/Drgona_SciML_UCSB.pptx
