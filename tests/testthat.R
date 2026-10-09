library(testthat)
library(rltoolsR)

heavy <- c(
  "ppo",
  "model-io-gpu"
)

on_cran <- !identical(Sys.getenv("NOT_CRAN"), "true")

test_dir <- if (dir.exists("testthat")) "testthat" else "tests/testthat"

if (on_cran) {
  message("--- RUNNING LIGHT TESTS ONLY ---")

  all_tests <- list.files(test_dir, pattern = "^test-.*\\.R$")
  all_names <- sub("^test-(.*)\\.R$", "\\1", all_tests)
  light_tests <- setdiff(all_names, heavy)
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
