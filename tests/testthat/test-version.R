test_that("rltools_version returns commit hash", {
  v <- rltools_version()
  expect_type(v, "character")
  expect_equal(nchar(v), 40L)
})
