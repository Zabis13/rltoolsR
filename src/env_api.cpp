// .Call interface to the environment contract.
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include "env.h"

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

namespace rltr {

Env* make_pendulum(int64_t seed);
Env* make_cartpole(int64_t seed);
Env* make_acrobot(int64_t seed);
Env* make_acrobot_swingup(int64_t seed);
Env* make_car(int64_t seed);
Env* make_car_track(int64_t seed);

Env* make_env(const std::string& name, int64_t seed) {
    if (name == "pendulum") return make_pendulum(seed);
    if (name == "cartpole") return make_cartpole(seed);
    if (name == "acrobot") return make_acrobot(seed);
    if (name == "acrobot_swingup") return make_acrobot_swingup(seed);
    if (name == "car") return make_car(seed);
    if (name == "car_track") return make_car_track(seed);
    return nullptr;
}

const char* const ENV_NAMES[] = {"pendulum", "cartpole", "acrobot", "acrobot_swingup", "car", "car_track"};
const int N_ENV_NAMES = sizeof(ENV_NAMES) / sizeof(ENV_NAMES[0]);

}  // namespace rltr

using rltr::Env;

// C++ exceptions must not cross R's longjmp: catch, copy the message, raise after unwinding.
// Rf_error() longjmps and skips C++ destructors, so every object with a
// non-trivial destructor (std::vector, std::string, ...) must live inside the
// RLTR_TRY block; outside it keep only plain values and PROTECTed SEXPs.
#define RLTR_TRY char rltr_err_[512] = ""; try {
#define RLTR_CATCH                                                                   \
    } catch (const std::exception& e) {                                              \
        std::snprintf(rltr_err_, sizeof(rltr_err_), "%s", e.what());                 \
    } catch (...) {                                                                  \
        std::snprintf(rltr_err_, sizeof(rltr_err_), "unknown C++ exception");        \
    }                                                                                \
    if (rltr_err_[0] != '\0') Rf_error("%s", rltr_err_);

static void env_finalizer(SEXP ptr) {
    Env* env = static_cast<Env*>(R_ExternalPtrAddr(ptr));
    delete env;
    R_ClearExternalPtr(ptr);
}

static SEXP spec_to_r(const rltr::EnvSpec& sp);

static Env* get_env(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("rltr_env")) {
        Rf_error("expected an environment pointer");
    }
    Env* env = static_cast<Env*>(R_ExternalPtrAddr(ptr));
    if (env == nullptr) Rf_error("environment pointer is NULL (freed or not restored after serialization)");
    return env;
}

static int64_t as_seed(SEXP seed) {
    if (Rf_isNull(seed)) return -1;
    double s = Rf_asReal(seed);
    if (ISNA(s) || s < 0) return -1;
    return static_cast<int64_t>(s);
}

static SEXP named_list(const char** names, int n) {
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    SEXP nms = PROTECT(Rf_allocVector(STRSXP, n));
    for (int i = 0; i < n; i++) SET_STRING_ELT(nms, i, Rf_mkChar(names[i]));
    Rf_setAttrib(out, R_NamesSymbol, nms);
    UNPROTECT(2);
    return out;
}

static SEXP int_vec(const std::vector<int>& v) {
    SEXP out = Rf_allocVector(INTSXP, v.size());
    for (size_t i = 0; i < v.size(); i++) INTEGER(out)[i] = v[i];
    return out;
}

static SEXP real_vec(const std::vector<double>& v) {
    SEXP out = Rf_allocVector(REALSXP, v.size());
    for (size_t i = 0; i < v.size(); i++) REAL(out)[i] = v[i];
    return out;
}

// Fills a REALSXP allocated before RLTR_TRY; call inside the try block.
static void copy_obs(const std::vector<float>& v, SEXP out) {
    if (static_cast<R_xlen_t>(v.size()) != XLENGTH(out)) {
        throw std::runtime_error("observation has length " + std::to_string(v.size()) +
                                 ", expected " + std::to_string(XLENGTH(out)));
    }
    for (size_t i = 0; i < v.size(); i++) REAL(out)[i] = v[i];
}

static int obs_dim(const Env* env) {
    return rltr::space_size(env->spec().observation_space);
}

static const char* space_type_name(rltr::SpaceType t) {
    switch (t) {
        case rltr::SpaceType::Box: return "box";
        case rltr::SpaceType::Discrete: return "discrete";
        case rltr::SpaceType::MultiDiscrete: return "multi_discrete";
        case rltr::SpaceType::MultiBinary: return "multi_binary";
    }
    return "unknown";
}

static const char* dtype_name(rltr::DType t) {
    switch (t) {
        case rltr::DType::Float32: return "float32";
        case rltr::DType::Int32: return "int32";
        case rltr::DType::UInt8: return "uint8";
    }
    return "unknown";
}

static const char* mode_name(rltr::EnvMode m) {
    switch (m) {
        case rltr::EnvMode::Single: return "single";
        case rltr::EnvMode::Sequential: return "sequential";
        case rltr::EnvMode::Simultaneous: return "simultaneous";
    }
    return "unknown";
}

// Fields match the R constructors space_box(), space_discrete(), ...
static SEXP space_to_r(const rltr::Space& s) {
    SEXP out;
    switch (s.type) {
        case rltr::SpaceType::Box: {
            const char* nms[] = {"type", "dtype", "shape", "low", "high"};
            out = PROTECT(named_list(nms, 5));
            SET_VECTOR_ELT(out, 2, int_vec(s.shape));
            SET_VECTOR_ELT(out, 3, real_vec(s.low));
            SET_VECTOR_ELT(out, 4, real_vec(s.high));
            break;
        }
        case rltr::SpaceType::Discrete: {
            const char* nms[] = {"type", "dtype", "n"};
            out = PROTECT(named_list(nms, 3));
            SET_VECTOR_ELT(out, 2, Rf_ScalarInteger(s.n));
            break;
        }
        case rltr::SpaceType::MultiDiscrete: {
            const char* nms[] = {"type", "dtype", "nvec"};
            out = PROTECT(named_list(nms, 3));
            SET_VECTOR_ELT(out, 2, int_vec(s.nvec));
            break;
        }
        default: {
            const char* nms[] = {"type", "dtype", "shape"};
            out = PROTECT(named_list(nms, 3));
            SET_VECTOR_ELT(out, 2, int_vec(s.shape));
            break;
        }
    }
    SET_VECTOR_ELT(out, 0, Rf_mkString(space_type_name(s.type)));
    SET_VECTOR_ELT(out, 1, Rf_mkString(dtype_name(s.dtype)));
    SEXP cls = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(cls, 0, Rf_mkChar((std::string("rl_space_") + space_type_name(s.type)).c_str()));
    SET_STRING_ELT(cls, 1, Rf_mkChar("rl_space"));
    Rf_setAttrib(out, R_ClassSymbol, cls);
    UNPROTECT(2);
    return out;
}

extern "C" SEXP rlt_env_names(void) {
    SEXP out = PROTECT(Rf_allocVector(STRSXP, rltr::N_ENV_NAMES));
    for (int i = 0; i < rltr::N_ENV_NAMES; i++) SET_STRING_ELT(out, i, Rf_mkChar(rltr::ENV_NAMES[i]));
    UNPROTECT(1);
    return out;
}

extern "C" SEXP rlt_env_make(SEXP name, SEXP seed) {
    const char* nm = CHAR(STRING_ELT(name, 0));
    int64_t s = as_seed(seed);
    Env* env = nullptr;
    RLTR_TRY
        env = rltr::make_env(nm, s);
        if (env == nullptr) throw std::invalid_argument(std::string("unknown environment: ") + nm);
    RLTR_CATCH
    SEXP ptr = PROTECT(R_MakeExternalPtr(env, Rf_install("rltr_env"), R_NilValue));
    R_RegisterCFinalizerEx(ptr, env_finalizer, TRUE);
    UNPROTECT(1);
    return ptr;
}

extern "C" SEXP rlt_env_spec(SEXP ptr) {
    return spec_to_r(get_env(ptr)->spec());
}

static SEXP spec_to_r(const rltr::EnvSpec& sp) {
    const char* nms[] = {"name", "mode", "n_players", "max_steps", "observation_space", "action_space"};
    SEXP out = PROTECT(named_list(nms, 6));
    SET_VECTOR_ELT(out, 0, Rf_mkString(sp.name.c_str()));
    SET_VECTOR_ELT(out, 1, Rf_mkString(mode_name(sp.mode)));
    SET_VECTOR_ELT(out, 2, Rf_ScalarInteger(sp.n_players));
    SET_VECTOR_ELT(out, 3, Rf_ScalarInteger(sp.max_steps));
    SET_VECTOR_ELT(out, 4, space_to_r(sp.observation_space));
    SET_VECTOR_ELT(out, 5, space_to_r(sp.action_space));
    UNPROTECT(1);
    return out;
}

extern "C" SEXP rlt_env_reset(SEXP ptr, SEXP seed) {
    Env* env = get_env(ptr);
    int64_t s = as_seed(seed);
    SEXP obs = PROTECT(Rf_allocVector(REALSXP, obs_dim(env)));
    RLTR_TRY
        copy_obs(env->reset(s), obs);
    RLTR_CATCH
    UNPROTECT(1);
    return obs;
}

extern "C" SEXP rlt_env_step(SEXP ptr, SEXP action) {
    Env* env = get_env(ptr);
    SEXP a = PROTECT(Rf_coerceVector(action, REALSXP));
    SEXP obs = PROTECT(Rf_allocVector(REALSXP, obs_dim(env)));
    double reward = 0;
    bool terminated = false, truncated = false;
    RLTR_TRY
        std::vector<double> actions(REAL(a), REAL(a) + XLENGTH(a));
        rltr::StepResult r = env->step(actions);
        copy_obs(r.obs, obs);
        reward = r.reward;
        terminated = r.terminated;
        truncated = r.truncated;
    RLTR_CATCH
    const char* nms[] = {"obs", "reward", "terminated", "truncated"};
    SEXP out = PROTECT(named_list(nms, 4));
    SET_VECTOR_ELT(out, 0, obs);
    SET_VECTOR_ELT(out, 1, Rf_ScalarReal(reward));
    SET_VECTOR_ELT(out, 2, Rf_ScalarLogical(terminated));
    SET_VECTOR_ELT(out, 3, Rf_ScalarLogical(truncated));
    UNPROTECT(3);
    return out;
}

// Two-phase results of unknown size (legal_mask(), save_state()): the C++
// vector must be gone before R allocates (an allocation error longjmps too).
// Phase 1 calls the method only to get the size; then the SEXP is allocated;
// phase 2 calls it again and copies straight into the SEXP. This relies on
// legal_mask() and save_state() being pure: const, no RNG, no clock, the same
// result on every call. A size change between the calls is reported as an error.
static void size_mismatch(const char* what, size_t first, size_t second) {
    Rf_error("%s returned %lu elements, then %lu: it must give the same result on every call",
             what, static_cast<unsigned long>(first), static_cast<unsigned long>(second));
}

// Copies a 0/1 mask into a LGLSXP of the same length; call inside RLTR_TRY.
static size_t copy_mask(const std::vector<uint8_t>& mask, SEXP out) {
    if (static_cast<R_xlen_t>(mask.size()) == XLENGTH(out)) {
        for (size_t i = 0; i < mask.size(); i++) LOGICAL(out)[i] = mask[i] != 0;
    }
    return mask.size();
}

extern "C" SEXP rlt_env_legal_mask(SEXP ptr) {
    Env* env = get_env(ptr);
    size_t n = 0, got = 0;
    {
        RLTR_TRY
            n = env->legal_mask().size();
        RLTR_CATCH
    }
    if (n == 0) return R_NilValue;  // no mask: every action is legal
    SEXP out = PROTECT(Rf_allocVector(LGLSXP, n));
    {
        RLTR_TRY
            got = copy_mask(env->legal_mask(), out);
        RLTR_CATCH
    }
    if (got != n) size_mismatch("legal_mask()", n, got);
    UNPROTECT(1);
    return out;
}

extern "C" SEXP rlt_env_save_state(SEXP ptr) {
    Env* env = get_env(ptr);
    size_t n = 0, got = 0;
    {
        RLTR_TRY
            n = env->save_state().size();
        RLTR_CATCH
    }
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, n));
    {
        RLTR_TRY
            std::vector<uint8_t> st = env->save_state();
            got = st.size();
            if (got == n && n > 0) std::memcpy(RAW(out), st.data(), n);
        RLTR_CATCH
    }
    if (got != n) size_mismatch("save_state()", n, got);
    UNPROTECT(1);
    return out;
}

extern "C" SEXP rlt_env_load_state(SEXP ptr, SEXP state) {
    Env* env = get_env(ptr);
    if (TYPEOF(state) != RAWSXP) Rf_error("state must be a raw vector");
    RLTR_TRY
        env->load_state(std::vector<uint8_t>(RAW(state), RAW(state) + XLENGTH(state)));
    RLTR_CATCH
    return R_NilValue;
}

// ---- Vectorized environments ----

using rltr::VecEnv;

static void vec_finalizer(SEXP ptr) {
    VecEnv* v = static_cast<VecEnv*>(R_ExternalPtrAddr(ptr));
    delete v;
    R_ClearExternalPtr(ptr);
}

static VecEnv* get_vec(SEXP ptr) {
    if (TYPEOF(ptr) != EXTPTRSXP || R_ExternalPtrTag(ptr) != Rf_install("rltr_vec_env")) {
        Rf_error("expected a vectorized environment pointer");
    }
    VecEnv* v = static_cast<VecEnv*>(R_ExternalPtrAddr(ptr));
    if (v == nullptr) Rf_error("environment pointer is NULL (freed or not restored after serialization)");
    return v;
}

extern "C" SEXP rlt_vec_make(SEXP name, SEXP n, SEXP seed) {
    const char* nm = CHAR(STRING_ELT(name, 0));
    int num = Rf_asInteger(n);
    int64_t s = as_seed(seed);
    VecEnv* v = nullptr;
    RLTR_TRY
        v = new VecEnv(nm, num, s);
    RLTR_CATCH
    SEXP ptr = PROTECT(R_MakeExternalPtr(v, Rf_install("rltr_vec_env"), R_NilValue));
    R_RegisterCFinalizerEx(ptr, vec_finalizer, TRUE);
    UNPROTECT(1);
    return ptr;
}

extern "C" SEXP rlt_vec_spec(SEXP ptr) {
    return spec_to_r(get_vec(ptr)->spec());
}

extern "C" SEXP rlt_vec_reset(SEXP ptr, SEXP seed) {
    VecEnv* v = get_vec(ptr);
    int64_t s = as_seed(seed);
    SEXP obs = PROTECT(Rf_allocMatrix(REALSXP, v->n(), v->obs_dim()));
    RLTR_TRY
        v->reset(s, REAL(obs));
    RLTR_CATCH
    UNPROTECT(1);
    return obs;
}

extern "C" SEXP rlt_vec_step(SEXP ptr, SEXP action) {
    VecEnv* v = get_vec(ptr);
    const int n = v->n();
    SEXP a = PROTECT(Rf_coerceVector(action, REALSXP));
    if (XLENGTH(a) != static_cast<R_xlen_t>(n) * v->act_dim()) {
        Rf_error("action must be a %d x %d matrix", n, v->act_dim());
    }
    const char* nms[] = {"obs", "reward", "terminated", "truncated", "final_obs"};
    SEXP out = PROTECT(named_list(nms, 5));
    SEXP obs = Rf_allocMatrix(REALSXP, n, v->obs_dim());
    SET_VECTOR_ELT(out, 0, obs);
    SEXP reward = Rf_allocVector(REALSXP, n);
    SET_VECTOR_ELT(out, 1, reward);
    SEXP term = Rf_allocVector(LGLSXP, n);
    SET_VECTOR_ELT(out, 2, term);
    SEXP trunc = Rf_allocVector(LGLSXP, n);
    SET_VECTOR_ELT(out, 3, trunc);
    SEXP final_obs = Rf_allocMatrix(REALSXP, n, v->obs_dim());
    SET_VECTOR_ELT(out, 4, final_obs);
    RLTR_TRY
        v->step(REAL(a), REAL(obs), REAL(reward), LOGICAL(term), LOGICAL(trunc), REAL(final_obs));
    RLTR_CATCH
    UNPROTECT(2);
    return out;
}

extern "C" SEXP rlt_vec_legal_mask(SEXP ptr) {
    VecEnv* v = get_vec(ptr);
    size_t n = 0, got = 0;
    {
        RLTR_TRY
            n = v->legal_mask().size();
        RLTR_CATCH
    }
    if (n == 0) return R_NilValue;  // no mask: every action is legal
    SEXP out = PROTECT(Rf_allocMatrix(LGLSXP, v->n(), static_cast<int>(n) / v->n()));
    {
        RLTR_TRY
            got = copy_mask(v->legal_mask(), out);
        RLTR_CATCH
    }
    if (got != n) size_mismatch("legal_mask()", n, got);
    UNPROTECT(1);
    return out;
}
