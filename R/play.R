#' Play tic-tac-toe against a model
#'
#' Interactive game in the console: you enter a cell `1..9` (row-major),
#' the model answers with its greedy move. Enter `q` to quit. Input is read
#' with [readline()] in an interactive session and from stdin under
#' `Rscript`.
#'
#' @param model Result of [rl_ppo_train()] or [rl_load_model()].
#' @param human Your side: `"O"` (model moves first) or `"X"`.
#' @param device Device for the game environment.
#' @return Invisibly, the winner (`"X"`, `"O"`), `"draw"`, or `NA` if quit.
#' @export
rl_play_tictactoe <- function(model, human = c("O", "X"), device = "cpu") {
  human <- match.arg(human)
  read_line <- function(prompt) {
    if (interactive()) return(readline(prompt))
    cat(prompt)
    x <- readLines("stdin", n = 1)
    if (length(x)) x else "q"
  }
  show <- function(b) {
    s <- ifelse(b > 0, "X", ifelse(b < 0, "O", "."))
    for (r in 0:2) cat(" ", s[r * 3 + 1:3], "\n")
    cat("\n")
  }

  g <- rl_gpu_env("tictactoe", n = 1, opponent = "self", device = device)
  obs <- rl_reset(g)
  legal <- rl_legal_actions(g)
  board <- numeric(9)                          # X = +1, O = -1
  side <- "X"
  cat("cells:\n  1 2 3\n  4 5 6\n  7 8 9\n\n")
  repeat {
    show(board)
    if (side == human) {
      repeat {
        x <- trimws(read_line(sprintf("your move (%s, 1-9, q): ", human)))
        if (x == "q") return(invisible(NA_character_))
        a <- suppressWarnings(as.integer(x)) - 1L
        if (!is.na(a) && a %in% 0:8 && legal[1, a + 1]) break
      }
    } else {
      a <- rl_act(model, obs, legal, greedy = TRUE)
      cat("model (", side, ") plays ", a + 1, "\n", sep = "")
    }
    board[a + 1] <- if (side == "X") 1 else -1
    r <- rl_step(g, a)
    if (r$terminated[1]) {
      show(board)
      winner <- switch(as.character(r$reward[1]),
                       "1" = side, "-1" = if (side == "X") "O" else "X", "draw")
      cat(if (winner == "draw") "draw" else
            if (winner == human) "you won" else "model won", "\n")
      return(invisible(winner))
    }
    obs <- r$obs
    legal <- r$legal
    side <- if (side == "X") "O" else "X"
  }
}
