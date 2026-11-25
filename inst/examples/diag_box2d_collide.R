# Box2D collision checks behind the lunar lander (rlt_box2d_check(),
# src/env_box2d.cpp): a body dropped from 10 m onto the ground. A body that fell
# through the ground ends far below 0. For the Linux CI failure where the lander
# falls through the terrain (test-box2d.R).
#
#   Rscript inst/examples/diag_box2d_collide.R
#
# Rows: a box on polygon ground, default filter (60 Hz, 4 sub-steps; the
# test's first check); then stepped as the lander (50 Hz, 8 sub-steps): a box on
# segment ground (as the lander's terrain); a box with the lander's filter on
# polygon ground; the same on segment ground; the lander's hull on segment
# ground; the hull with its legs (on the ground: between 0.3 and 1).

suppressMessages(devtools::load_all(quiet = TRUE))
r <- .Call(rltoolsR:::rlt_box2d_check)
cat(sprintf("Box2D %d.%d.%d\n\n", r[1], r[2], r[3]))

# where each Box2D function resolves (dladdr): all should be this package's library
s <- .Call(rltoolsR:::rlt_box2d_symbols)
cat("Symbol binding:\n")
cat(sprintf("  %-24s %s\n", names(s), s), sep = "")
cat("\n")

d <- data.frame(setup = c("box, polygon ground, default filter",
                          "box, segment ground, default filter",
                          "box, polygon ground, lander filter",
                          "box, segment ground, lander filter",
                          "lander hull, segment ground",
                          "lander hull + legs, segment ground"),
                y = r[4:9],
                expected = c(rep("0.5", 4), "0.333", "0.3..1"))
d$ok <- c(abs(r[4:7] - 0.5) < 0.02, abs(r[8] - 10 / 30) < 0.02, r[9] > 0.3 && r[9] < 1)
print(d, row.names = FALSE, digits = 4)

cat("\nPolygon-segment stages:\n")
cat(sprintf("  narrow phase: %g points (2), separation %.4f (-0.1)\n", r[10], r[11]))
cat(sprintf("  segment bounds: lower (%g, %g), upper (%g, %g)  ((-50, 0), (50, 0))\n",
            r[12], r[13], r[14], r[15]))
cat(sprintf("  broad phase: at most %g contacts during the fall (>= 1)\n", r[16]))

cat("\nSegment bounds, step by step:\n")
cat(sprintf("  b2ComputeSegmentBounds, identity: (%g, %g) - (%g, %g)  ((-50, 0) - (50, 0))\n",
            r[17], r[18], r[19], r[20]))
cat(sprintf("  ground body transform: p (%g, %g), q (%g, %g)  ((0, 0), (1, 0))\n",
            r[21], r[22], r[23], r[24]))
cat(sprintf("  stored segment: (%g, %g) - (%g, %g)  ((-50, 0) - (50, 0))\n",
            r[25], r[26], r[27], r[28]))
cat(sprintf("  b2ComputeSegmentBounds, body transform: (%g, %g) - (%g, %g)  ((-50, 0) - (50, 0))\n",
            r[38], r[39], r[40], r[41]))
cat(sprintf("  recomputed as at creation, margin 0: (%g, %g) - (%g, %g)  ((-50, 0) - (50, 0))\n",
            r[42], r[43], r[44], r[45]))
cat(sprintf("  recomputed as at creation, margin 0.02: (%g, %g) - (%g, %g)  ((-50.02, -0.02) - (50.02, 0.02))\n",
            r[46], r[47], r[48], r[49]))
cat(sprintf("  broad-phase tree near the origin: %g shape(s) (1)\n", r[29]))
cat(sprintf("  fat AABB (broad-phase pairs): (%g, %g) - (%g, %g)  (about (-50, 0) - (50, 0))\n",
            r[30], r[31], r[32], r[33]))
cat(sprintf("  tree node AABB: (%g, %g) - (%g, %g)  (about (-50, 0) - (50, 0))\n",
            r[34], r[35], r[36], r[37]))
