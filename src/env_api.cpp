// .Call interface to the environment contract.
#include <R.h>
#include <Rinternals.h>

#include <cstdio>
#include <exception>

#include "env.h"

namespace rltr {

Env* make_pendulum(int64_t seed);
Env* make_acrobot(int64_t seed);
Env* make_acrobot_swingup(int64_t seed);
Env* make_car(int64_t seed);
Env* make_car_track(int64_t seed);

Env* make_env(const std::string& name, int64_t seed) {
    if (name == "pendulum") return make_pendulum(seed);
    if (name == "acrobot") return make_acrobot(seed);
    if (name == "acrobot_swingup") return make_acrobot_swingup(seed);
    if (name == "car") return make_car(seed);
    if (name == "car_track") return make_car_track(seed);
    return nullptr;
}

std::vector<std::string> env_names() {
    return {"pendulum", "acrobot", "acrobot_swingup", "car", "car_track"};
}

}  // namespace rltr

using rltr::Env;

// C++ exceptions must not cross R's longjmp: catch, copy the message, raise after unwinding.
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

static SEXP obs_vec(const std::vector<float>& v) {
    SEXP out = Rf_allocVector(REALSXP, v.size());
    for (size_t i = 0; i < v.size(); i++) REAL(out)[i] = v[i];
    return out;
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
    std::vector<std::string> names = rltr::env_names();
    SEXP out = PROTECT(Rf_allocVector(STRSXP, names.size()));
    for (size_t i = 0; i < names.size(); i++) SET_STRING_ELT(out, i, Rf_mkChar(names[i].c_str()));
    UNPROTECT(1);
    return out;
}

extern "C" SEXP rlt_env_make(SEXP name, SEXP seed) {
    std::string nm = CHAR(STRING_ELT(name, 0));
    int64_t s = as_seed(seed);
    Env* env = nullptr;
    RLTR_TRY
        env = rltr::make_env(nm, s);
        if (env == nullptr) throw std::invalid_argument("unknown environment: " + nm);
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
    std::vector<float> obs;
    RLTR_TRY
        obs = env->reset(s);
    RLTR_CATCH
    return obs_vec(obs);
}

extern "C" SEXP rlt_env_step(SEXP ptr, SEXP action) {
    Env* env = get_env(ptr);
    SEXP a = PROTECT(Rf_coerceVector(action, REALSXP));
    std::vector<double> actions(REAL(a), REAL(a) + XLENGTH(a));
    UNPROTECT(1);
    rltr::StepResult r;
    RLTR_TRY
        r = env->step(actions);
    RLTR_CATCH
    const char* nms[] = {"obs", "reward", "terminated", "truncated"};
    SEXP out = PROTECT(named_list(nms, 4));
    SET_VECTOR_ELT(out, 0, obs_vec(r.obs));
    SET_VECTOR_ELT(out, 1, Rf_ScalarReal(r.reward));
    SET_VECTOR_ELT(out, 2, Rf_ScalarLogical(r.terminated));
    SET_VECTOR_ELT(out, 3, Rf_ScalarLogical(r.truncated));
    UNPROTECT(1);
    return out;
}

extern "C" SEXP rlt_env_legal_mask(SEXP ptr) {
    Env* env = get_env(ptr);
    std::vector<uint8_t> mask = env->legal_mask();
    if (mask.empty()) return R_NilValue;
    SEXP out = PROTECT(Rf_allocVector(LGLSXP, mask.size()));
    for (size_t i = 0; i < mask.size(); i++) LOGICAL(out)[i] = mask[i] != 0;
    UNPROTECT(1);
    return out;
}

extern "C" SEXP rlt_env_save_state(SEXP ptr) {
    Env* env = get_env(ptr);
    std::vector<uint8_t> st;
    RLTR_TRY
        st = env->save_state();
    RLTR_CATCH
    SEXP out = PROTECT(Rf_allocVector(RAWSXP, st.size()));
    if (!st.empty()) std::memcpy(RAW(out), st.data(), st.size());
    UNPROTECT(1);
    return out;
}

extern "C" SEXP rlt_env_load_state(SEXP ptr, SEXP state) {
    Env* env = get_env(ptr);
    if (TYPEOF(state) != RAWSXP) Rf_error("state must be a raw vector");
    std::vector<uint8_t> st(RAW(state), RAW(state) + XLENGTH(state));
    RLTR_TRY
        env->load_state(st);
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
    std::string nm = CHAR(STRING_ELT(name, 0));
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
    std::vector<uint8_t> mask = v->legal_mask();
    if (mask.empty()) return R_NilValue;
    SEXP out = PROTECT(Rf_allocMatrix(LGLSXP, v->n(), static_cast<int>(mask.size()) / v->n()));
    for (size_t i = 0; i < mask.size(); i++) LOGICAL(out)[i] = mask[i] != 0;
    UNPROTECT(1);
    return out;
}
