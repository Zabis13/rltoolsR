## Resubmission

This is a resubmission. In this version I have:

* Removed the redundant "toolkit for R" from the start of the Description.

* R/log.R: the only `cat()` call there wrote rows to the CSV log file
  (`file = ...`), not to the console. It is replaced by `writeLines()` on a
  file connection. The remaining console output in the package is in
  print() methods and the interactive `rl_play_tictactoe()`; informational output
  uses `message()`/`warning()`.

* Authors and copyright holders: I searched all bundled sources
  (`inst/include`, `src`) for third-party code and notices, and:
  - removed the unused 'wasm32' headers of 'RLtools' (one of them
    contained cbrt code from musl / FreeBSD msun);
  - added to `Authors@R` all authors and copyright holders of the code that
    remains: Jonas Eschmann ('RLtools'); Rich Felker ('musl'),
    Sun Microsystems, Inc. and Ian Lance Taylor (acos code in
    `inst/include/rl_tools/math/operations_generic.h`); OpenAI and
    Farama Foundation ('Gymnasium'), and the RLPy authors Christoph Dann,
    Alborz Geramifard, Robert H. Klein, William Dabney and Jonathan P. How
    (acrobot environment adapted in `inst/include/rl_tools/rl/environments/acrobot/`);
  - listed every component with its copyright notice and full license text
    in `inst/COPYRIGHTS`. The original notices in the bundled files are
    preserved unchanged.

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* local: Ubuntu 24.04, R 4.4.x, GCC 14
