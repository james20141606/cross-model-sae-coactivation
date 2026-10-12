# PREREGISTRATION: a leakage-controlled Procrustes-aligned pairing baseline

Committed BEFORE the pairing is implemented and before any Procrustes quantity exists.
Git history is the timestamp.

## Why

arXiv:2605.24577 ("Polymorphism Is Rotation") reports that decoder-cosine similarity matches
across seeds at 98% while an SAE trained on one seed reconstructs another's activations at
negative explained variance, and argues the standard universality metric is blind to that
failure. Our answer so far is that survival under lexical residualization holds under TWO
pairing rules (mutual-NN and the transplanted Lan argmax). A rotation account predicts a
third thing: if the correspondence is a rotation of a shared subspace, then pairing in an
ORTHOGONALLY ALIGNED coordinate system should find it at least as well as either raw rule.
arXiv:2607.08499 makes the same prediction from the method side (Procrustes pre-alignment
produces more universal features than post-hoc pairing). This registers that baseline.

## Design

**Alignment, fitted on a disjoint document split.** Documents are split 50/50 by a fixed
seed. On the TRAIN half only: reduce each model's pooled feature matrix to its top-r left
singular basis (r = 64, fixed here, not tuned); fit the orthogonal Q that best aligns model
B's document basis to model A's (Q = U V^T from the SVD of Ub^T Ua -- closed-form orthogonal
Procrustes over the SHARED document axis, so no feature correspondence is assumed).
Feature coordinates are the projections of each feature's document vector onto those bases;
B's are mapped through Q. Pairs are mutual nearest neighbours by cosine in that aligned
coordinate space, computed on the TRAIN half.

**Evaluation, on the held-out half.** The pairs are frozen and the identical lexical
residualization runs on the TEST half only -- same adequacy gate, same shuffled-design
calibration, same within-fold null, same n_perm, same governing-k rule. Nothing about the
alignment is refit after the split, and no test-half quantity enters the fit.

## Registered readout and branches

R_lex(procrustes) at the largest adequacy-passing k, alongside the two committed rules
recomputed on the SAME held-out half so the three are comparable.

- **P1 alignment does not rescue**: R_lex(procrustes) >= 0.70 in >= 2 of 3 cells. Survival
  is not an artifact of unaligned pairing; the rotation account's prediction is met AND the
  lexical control still finds the excess intact. Strengthens the existing claim.
- **P2 alignment collapses**: R_lex(procrustes) < 0.30 in >= 2 of 3 cells. The excess that
  survives residualization under raw pairing does NOT survive it under aligned pairing --
  a genuine dissociation, reported as such, and the paper's claim narrows to the raw rules.
- **P3 partial**: otherwise.
- **P4 not testable**: adequacy fails at every k on >= 2 of 3 cells on the held-out half
  (halving the documents halves the first stage's power; this is the anticipated failure).

Reader: code/analysis/procrustes_summary.py, committed with the implementation, refuses
partial input, prints the branch verbatim and the candidate k set it saw.

**Bound stated now**: r = 64 and the 50/50 split are fixed a priori, not swept. If P4 fires
we report the arm as underpowered on half the corpus rather than enlarging the corpus to
rescue it -- that would be an adaptive stopping rule on the same statistic.

No Procrustes quantity exists at the time of this registration.
