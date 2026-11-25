# Box2D collision checks behind the lunar lander (rlt_box2d_check(),
# src/env_box2d.cpp): a body dropped from 10 m onto the ground, and a segment
# shape's bounding box. A body that fell through the ground ends far below 0.
# gcc 15 for x86-64-v3 (Ubuntu 26.04's default) miscompiled Box2D's bounding
# boxes without -fno-tree-slp-vectorize (see ./configure): segment bounds came
# out as (2500, 0)-(0, 0) and bodies fell through segment ground.
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

cat(sprintf("\nsegment bounds: (%g, %g) - (%g, %g)  (about (-50, 0) - (50, 0))\n",
            r[10], r[11], r[12], r[13]))
