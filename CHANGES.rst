Changes
-------

0.3.0 (2021-10-30)
------------------
* Revert for Drop RAdam.

0.2.0 (2021-10-25)
------------------
* Drop RAdam optimizer since it is included in pytorch.
* Do not include tests as installable package.
* Preserver memory layout where possible.
* Add MADGRAD optimizer.

0.1.0 (2021-01-01)
------------------
* Initial release.
* Added support for A2GradExp, A2GradInc, A2GradUni, AccSGD, AdaBelief,
  AdaBound, AdaMod, Adafactor, Adahessian, AdamP, AggMo, Apollo,
  DiffGrad, Lamb, Lookahead, NovoGrad, PID, QHAdam, QHM, RAdam, Ranger,
  RangerQH, RangerVA, SGDP, SGDW, SWATS, Shampoo, Yogi.
