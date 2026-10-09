# Independent reference for the discrete policy's masked log-softmax, for tests
# only: plain R in double precision (subtract the column max, log-sum-exp). The
# package computes the same thing with ggmlR::ag_log_softmax() (policy_logp());
# checking that against itself would not catch an error in it.
ref_masked_log_softmax <- function(logits, legal) {
  logits <- logits + rltoolsR:::PPO_MASK * (1 - legal)
  m <- apply(logits, 2, max)
  z <- sweep(logits, 2, m)
  sweep(z, 2, log(colSums(exp(z))))
}
