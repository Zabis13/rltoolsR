## Resubmission

This is a resubmission. In this version I have:

* Removed the redundant "toolkit for R" from the start of the Description.

* R/log.R: the only `cat()` call there wrote rows to the CSV log file
  (`file = ...`), not to the console. It is replaced by `writeLines()` on a
  file connection. The remaining console output in the package is in
  print() methods and the interactive `rl_play_tictactoe()`; informational output
  uses `message()`/`warning()`.

* Authors and copyright holders: the package no longer bundles the
  'RLtools' headers (`inst/include` is removed). The five continuous-control
  environments that were used from it are ported to standalone C++ in
  `src/env_control.cpp`, with the default race track in
  `src/car_track_data.h`. As a result:
  - the 'musl' / FreeBSD msun acos code is no longer included, and Rich
    Felker, Sun Microsystems, Inc. and Ian Lance Taylor are removed from
    `Authors@R`;
  - `Authors@R` lists the authors and copyright holders of the code the
    ported environments derive from: Jonas Eschmann ('RLtools'); OpenAI and
    Farama Foundation ('Gymnasium'), and the RLPy authors Christoph Dann,
    Alborz Geramifard, Robert H. Klein, William Dabney and Jonathan P. How
    (acrobot environment);
  - `inst/COPYRIGHTS` gives each of these components with its copyright
    notice and full license text, and names the files that use them.

* Removed the exported `rltools_version()`, which returned the commit hash
  of the bundled 'RLtools' headers.

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* local: Ubuntu 24.04, R 4.4.x, GCC 14
