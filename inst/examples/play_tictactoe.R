# Play tic-tac-toe against a saved model in the console.
#
#   Rscript inst/examples/play_tictactoe.R [model.rds] [O|X]
#
# The model is read from /tmp (only the file name is taken from the
# argument). Default side is O, so the model moves first.

library(rltoolsR)

args <- commandArgs(trailingOnly = TRUE)
path <- file.path("/tmp", basename(if (length(args) >= 1) args[1] else "tictactoe_ppo.rds"))
human <- if (length(args) >= 2) toupper(args[2]) else "O"

m <- rl_load_model(path, device = "cpu")
rl_play_tictactoe(m, human = human, device = "cpu")
