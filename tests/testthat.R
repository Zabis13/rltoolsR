library(testthat)
library(rltoolsR)

heavy <- c(
  "ppo",
  "model-io-gpu",
  "env-rltools"
)

# Under valgrind everything runs 20-50x slower, and the PPO training tests make
# the memtest take ~35 min. valgrind is here to catch C-level memory bugs in the
# package's own C++ (the environments in src/); the ggml bindings are covered by
# ggmlR's own valgrind run. So under valgrind run only the training-free tests.
valgrind_core <- c(
  "spaces",
  "vec1",
  "vec-env",
  "env-pendulum",
  "env-rltools",   # all C++ environments in src/
  "env-errors",    # C++ errors must not leak past Rf_error()
  "replay-buffer"
)

# Detect valgrind via several independent signals (any one is enough):
#  1. the RUNNING_ON_VALGRIND env var some setups export
#  2. vgpreload_*.so injected into the process (visible in /proc/self/maps)
#  3. valgrind in the command line of the parent process
detect_valgrind <- function() {
  if (nzchar(Sys.getenv("RUNNING_ON_VALGRIND"))) return(TRUE)
  m <- tryCatch(readLines("/proc/self/maps", warn = FALSE),
                error = function(e) character(0))
  if (any(grepl("valgrind|vgpreload", m, ignore.case = TRUE))) return(TRUE)
  s <- tryCatch(readLines("/proc/self/status", warn = FALSE),
                error = function(e) character(0))
  ppid <- sub("^PPid:\\s*", "", grep("^PPid:", s, value = TRUE))
  if (length(ppid) == 1L) {
    pcmd <- tryCatch(
      readChar(file.path("/proc", ppid, "cmdline"), 1e4, useBytes = TRUE),
      error = function(e) "")
    if (grepl("valgrind", pcmd, ignore.case = TRUE)) return(TRUE)
  }
  FALSE
}
under_valgrind <- isTRUE(detect_valgrind())
message("valgrind detected: ", under_valgrind)

# Thread budget: 1 under valgrind (OpenMP worker TLS shows up as "possibly
# lost"), otherwise 2 as CRAN policy requires. Sys.setenv() alone is not enough
# once libgomp has sized its pool, so also pin it via ggml_set_n_threads().
test_threads <- if (under_valgrind) 1L else 2L
Sys.setenv(OMP_NUM_THREADS = as.character(test_threads))
ggmlR::ggml_set_n_threads(test_threads)

on_cran <- !identical(Sys.getenv("NOT_CRAN"), "true")

test_dir <- if (dir.exists("testthat")) "testthat" else "tests/testthat"

if (on_cran) {
  message("--- RUNNING LIGHT TESTS ONLY ---")

  all_tests <- list.files(test_dir, pattern = "^test-.*\\.R$")
  all_names <- sub("^test-(.*)\\.R$", "\\1", all_tests)
  light_tests <- if (under_valgrind) {
    message("--- VALGRIND DETECTED: running small core suite only ---")
    intersect(valgrind_core, all_names)
  } else {
    setdiff(all_names, heavy)
  }
  message("Tests to run: ", paste(light_tests, collapse = ", "))

  if (length(light_tests) == 0) {
    # fallback
    test_check("rltoolsR")
  } else {
    # testthat applies `filter` as grepl(filter, <test name>) with no anchors,
    # so anchor each name with ^...$ to avoid partial matches.
    filter_regex <- paste0("^(", paste(light_tests, collapse = "|"), ")$")
    test_check("rltoolsR", filter = filter_regex)
  }
} else {
  test_check("rltoolsR")
}
