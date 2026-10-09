#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include <rl_tools/version.h>

extern "C" SEXP rlt_version(void) {
    return Rf_mkString(RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH));
}

extern "C" {
SEXP rlt_env_names(void);
SEXP rlt_env_make(SEXP, SEXP);
SEXP rlt_env_spec(SEXP);
SEXP rlt_env_reset(SEXP, SEXP);
SEXP rlt_env_step(SEXP, SEXP);
SEXP rlt_env_legal_mask(SEXP);
SEXP rlt_env_save_state(SEXP);
SEXP rlt_env_load_state(SEXP, SEXP);
SEXP rlt_vec_make(SEXP, SEXP, SEXP);
SEXP rlt_vec_spec(SEXP);
SEXP rlt_vec_reset(SEXP, SEXP);
SEXP rlt_vec_step(SEXP, SEXP);
SEXP rlt_vec_legal_mask(SEXP);
}

static const R_CallMethodDef CallEntries[] = {
    {"rlt_version", (DL_FUNC) &rlt_version, 0},
    {"rlt_env_names", (DL_FUNC) &rlt_env_names, 0},
    {"rlt_env_make", (DL_FUNC) &rlt_env_make, 2},
    {"rlt_env_spec", (DL_FUNC) &rlt_env_spec, 1},
    {"rlt_env_reset", (DL_FUNC) &rlt_env_reset, 2},
    {"rlt_env_step", (DL_FUNC) &rlt_env_step, 2},
    {"rlt_env_legal_mask", (DL_FUNC) &rlt_env_legal_mask, 1},
    {"rlt_env_save_state", (DL_FUNC) &rlt_env_save_state, 1},
    {"rlt_env_load_state", (DL_FUNC) &rlt_env_load_state, 2},
    {"rlt_vec_make", (DL_FUNC) &rlt_vec_make, 3},
    {"rlt_vec_spec", (DL_FUNC) &rlt_vec_spec, 1},
    {"rlt_vec_reset", (DL_FUNC) &rlt_vec_reset, 2},
    {"rlt_vec_step", (DL_FUNC) &rlt_vec_step, 2},
    {"rlt_vec_legal_mask", (DL_FUNC) &rlt_vec_legal_mask, 1},
    {NULL, NULL, 0}
};

extern "C" void R_init_rltoolsR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
