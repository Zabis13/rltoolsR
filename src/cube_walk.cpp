// Cube scrambles on the host: random walks from solved and the sticker colours
// they lead to. The R versions (cube_walk_actions_r(), cube_host_scramble_r()
// in R/gpu_cube.R) are the reference; this is the same walk rule and the same
// move application, moved out of R because a truncation of all 256 games spent
// ~4.4 ms there (one runif(k * M) and max.col per step, a cbind index per move).
//
// Ported from cayleyR's generate_walk_cpp (src/generate_walk.cpp, same author):
// the no_undo / no_cancel scan and the apply rule new[i] = old[p[i]], with the
// group object replaced by the perms matrix rltoolsR already keeps.
//
// Random numbers: one unif_rand() per walk and step, an index into that
// step's allowed moves -- uniform over them, as the R version, but a different
// stream, so the same seed gives other scrambles than the R code did.

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <vector>

namespace {

// colours (k x S, column-major) after applying the moves of `acts`
// (k x depth, 0-based, column-major) to solved. perms: nrow x S, 1-based.
void apply_moves(const int* perms, int nrow, int S, const int* acts, int k,
                 int depth, int* col) {
  const int per_face = S / 6;
  std::vector<int> cur(S), nxt(S);
  for (int g = 0; g < k; ++g) {
    for (int i = 0; i < S; ++i) cur[i] = i / per_face;
    for (int t = 0; t < depth; ++t) {
      const int a = acts[g + (size_t) t * k];
      for (int i = 0; i < S; ++i) nxt[i] = cur[perms[a + (size_t) i * nrow] - 1];
      cur.swap(nxt);
    }
    for (int i = 0; i < S; ++i) col[g + (size_t) i * k] = cur[i];
  }
}

}  // namespace

extern "C" {

// k walks of `depth` moves over moves 0..n_moves-1 (k x depth, 0-based).
// commute (n_moves x n_moves logical) and order (n_moves int) NULL: no_undo,
// the inverse (a xor 1) of the last move is refused. Given: cayleyR's
// no_cancel -- scan the tail back while it commutes with the candidate; its
// inverse bans it, and so does the candidate itself met so often that one
// more would be shorter the other way (2 (cnt + 1) > order).
// colors TRUE also returns the colours (k x S) the walks lead to from solved.
SEXP rlt_cube_walk(SEXP perms, SEXP k_, SEXP depth_, SEXP n_moves_,
                   SEXP commute, SEXP order, SEXP colors) {
  const int k = Rf_asInteger(k_), depth = Rf_asInteger(depth_);
  const int M = Rf_asInteger(n_moves_);
  if (k == NA_INTEGER || k < 0 || depth == NA_INTEGER || depth < 0 ||
      M == NA_INTEGER || M < 2)
    Rf_error("rlt_cube_walk: bad k, depth or n_moves");
  const bool no_cancel = !Rf_isNull(commute);
  const int* comm = nullptr;
  const int* ord = nullptr;
  if (no_cancel) {
    if (!Rf_isLogical(commute) || Rf_length(commute) != M * M ||
        !Rf_isInteger(order) || Rf_length(order) != M)
      Rf_error("rlt_cube_walk: commute must be %d x %d logical, order %d integers", M, M, M);
    comm = LOGICAL(commute);
    ord = INTEGER(order);
  }

  SEXP acts = PROTECT(Rf_allocMatrix(INTSXP, k, depth));
  int* a = INTEGER(acts);
  std::vector<int> allowed(M);
  std::vector<char> active(M), ban(M);
  std::vector<int> cnt(M);

  GetRNGstate();
  for (int t = 0; t < depth; ++t) {
    for (int g = 0; g < k; ++g) {
      int n_ok = 0;
      if (t == 0) {
        for (int c = 0; c < M; ++c) allowed[n_ok++] = c;
      } else if (!no_cancel) {
        const int inv = a[g + (size_t) (t - 1) * k] ^ 1;
        for (int c = 0; c < M; ++c) if (c != inv) allowed[n_ok++] = c;
      } else {
        for (int c = 0; c < M; ++c) { active[c] = 1; ban[c] = 0; cnt[c] = 0; }
        for (int j = t - 1; j >= 0; --j) {
          const int b = a[g + (size_t) j * k];
          bool any = false;
          for (int c = 0; c < M; ++c) {
            if (!active[c]) continue;
            if (b == (c ^ 1)) { ban[c] = 1; active[c] = 0; continue; }
            if (!comm[c + (size_t) b * M]) { active[c] = 0; continue; }
            if (b == c) ++cnt[c];
            any = true;
          }
          if (!any) break;
        }
        for (int c = 0; c < M; ++c) {
          if (ord[c] > 0 && 2 * (cnt[c] + 1) > ord[c]) ban[c] = 1;
          if (!ban[c]) allowed[n_ok++] = c;
        }
      }
      if (n_ok == 0) {
        PutRNGstate();
        UNPROTECT(1);
        Rf_error("no move left to take at step %d", t + 1);
      }
      int r = (int) (unif_rand() * n_ok);
      if (r >= n_ok) r = n_ok - 1;
      a[g + (size_t) t * k] = allowed[r];
    }
  }
  PutRNGstate();

  SEXP col = R_NilValue;
  if (Rf_asLogical(colors) == TRUE) {
    SEXP dim = Rf_getAttrib(perms, R_DimSymbol);
    if (!Rf_isInteger(perms) || Rf_length(dim) != 2)
      Rf_error("rlt_cube_walk: perms must be an integer matrix");
    const int nrow = INTEGER(dim)[0], S = INTEGER(dim)[1];
    if (nrow < M || S % 6 != 0) Rf_error("rlt_cube_walk: perms does not fit n_moves");
    col = PROTECT(Rf_allocMatrix(INTSXP, k, S));
    apply_moves(INTEGER(perms), nrow, S, a, k, depth, INTEGER(col));
    UNPROTECT(1);
  }
  PROTECT(col);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, acts);
  SET_VECTOR_ELT(out, 1, col);
  UNPROTECT(3);
  return out;
}

// colours (k x S) after the moves of `acts` (k x depth, 0-based; a move index
// may point at any row of perms, e.g. the identity row) from solved.
SEXP rlt_cube_apply(SEXP perms, SEXP acts) {
  SEXP pdim = Rf_getAttrib(perms, R_DimSymbol), adim = Rf_getAttrib(acts, R_DimSymbol);
  if (!Rf_isInteger(perms) || Rf_length(pdim) != 2 ||
      !Rf_isInteger(acts) || Rf_length(adim) != 2)
    Rf_error("rlt_cube_apply: perms and acts must be integer matrices");
  const int nrow = INTEGER(pdim)[0], S = INTEGER(pdim)[1];
  const int k = INTEGER(adim)[0], depth = INTEGER(adim)[1];
  if (S % 6 != 0) Rf_error("rlt_cube_apply: the number of stickers must be a multiple of 6");
  const int* a = INTEGER(acts);
  for (R_xlen_t i = 0; i < (R_xlen_t) k * depth; ++i)
    if (a[i] == NA_INTEGER || a[i] < 0 || a[i] >= nrow)
      Rf_error("rlt_cube_apply: move index out of range");
  SEXP col = PROTECT(Rf_allocMatrix(INTSXP, k, S));
  apply_moves(INTEGER(perms), nrow, S, a, k, depth, INTEGER(col));
  UNPROTECT(1);
  return col;
}

}  // extern "C"
