// Environment contract shared by all backends (RLtools, OpenSpiel, MAME, own simulators).
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rltr {

enum class SpaceType { Box, Discrete, MultiDiscrete, MultiBinary };
enum class DType { Float32, Int32, UInt8 };
enum class EnvMode { Single, Sequential, Simultaneous };

// Gymnasium-style space. Box uses shape/low/high, Discrete uses n,
// MultiDiscrete uses nvec, MultiBinary uses shape. Data is row-major.
struct Space {
    SpaceType type = SpaceType::Box;
    DType dtype = DType::Float32;
    std::vector<int> shape;
    std::vector<double> low;
    std::vector<double> high;
    int n = 0;
    std::vector<int> nvec;
};

struct EnvSpec {
    std::string name;
    EnvMode mode = EnvMode::Single;
    int n_players = 1;
    int max_steps = 0;  // 0 = no step limit
    Space observation_space;
    Space action_space;
};

struct StepResult {
    std::vector<float> obs;
    double reward = 0;
    bool terminated = false;
    bool truncated = false;
};

class Env {
public:
    virtual ~Env() = default;

    virtual const EnvSpec& spec() const = 0;

    // Negative seed keeps the current RNG stream.
    std::vector<float> reset(int64_t seed) {
        step_count_ = 0;
        return do_reset(seed);
    }

    StepResult step(const std::vector<double>& actions) {
        StepResult r = do_step(actions);
        step_count_++;
        int limit = spec().max_steps;
        r.truncated = !r.terminated && limit > 0 && step_count_ >= limit;
        return r;
    }

    // Empty for environments without discrete legal actions.
    virtual std::vector<uint8_t> legal_mask() const { return {}; }

    // Snapshot = step counter + environment payload. RNG state is not included.
    std::vector<uint8_t> save_state() const {
        std::vector<uint8_t> payload = do_save_state();
        std::vector<uint8_t> out(sizeof(step_count_) + payload.size());
        std::memcpy(out.data(), &step_count_, sizeof(step_count_));
        std::memcpy(out.data() + sizeof(step_count_), payload.data(), payload.size());
        return out;
    }

    void load_state(const std::vector<uint8_t>& state) {
        if (state.size() < sizeof(step_count_)) throw std::runtime_error("state snapshot is too short");
        std::memcpy(&step_count_, state.data(), sizeof(step_count_));
        do_load_state(std::vector<uint8_t>(state.begin() + sizeof(step_count_), state.end()));
    }

    int step_count() const { return step_count_; }

protected:
    virtual std::vector<float> do_reset(int64_t seed) = 0;
    virtual StepResult do_step(const std::vector<double>& actions) = 0;
    virtual std::vector<uint8_t> do_save_state() const = 0;
    virtual void do_load_state(const std::vector<uint8_t>& payload) = 0;

    int step_count_ = 0;
};

// Returns nullptr for an unknown name.
Env* make_env(const std::string& name, int64_t seed);
std::vector<std::string> env_names();

// Number of scalars in a flat sample of the space.
inline int space_size(const Space& s) {
    switch (s.type) {
        case SpaceType::Discrete: return 1;
        case SpaceType::MultiDiscrete: return static_cast<int>(s.nvec.size());
        default: {
            int n = 1;
            for (int d : s.shape) n *= d;
            return n;
        }
    }
}

// N copies of one environment stepped together. Finished copies are reset in
// the same step (Gymnasium "same-step" autoreset); their last observation is
// kept in final_obs. Arrays are column-major n x dim, matching R matrices.
class VecEnv {
public:
    VecEnv(const std::string& name, int n, int64_t seed) {
        if (n < 1) throw std::invalid_argument("n must be >= 1");
        for (int i = 0; i < n; i++) {
            Env* e = make_env(name, seed < 0 ? -1 : seed + i);
            if (e == nullptr) throw std::invalid_argument("unknown environment: " + name);
            envs_.emplace_back(e);
        }
        obs_dim_ = space_size(spec().observation_space);
        act_dim_ = space_size(spec().action_space);
    }

    const EnvSpec& spec() const { return envs_[0]->spec(); }
    int n() const { return static_cast<int>(envs_.size()); }
    int obs_dim() const { return obs_dim_; }
    int act_dim() const { return act_dim_; }

    void reset(int64_t seed, double* obs) {
        for (int i = 0; i < n(); i++) {
            put(obs, i, envs_[i]->reset(seed < 0 ? -1 : seed + i));
        }
    }

    // actions: n x act_dim; outputs sized by the caller.
    void step(const double* actions, double* obs, double* reward, int* terminated, int* truncated,
              double* final_obs) {
        std::vector<double> a(act_dim_);
        for (int i = 0; i < n(); i++) {
            for (int j = 0; j < act_dim_; j++) a[j] = actions[i + j * n()];
            StepResult r = envs_[i]->step(a);
            reward[i] = r.reward;
            terminated[i] = r.terminated;
            truncated[i] = r.truncated;
            put(final_obs, i, r.obs);
            put(obs, i, (r.terminated || r.truncated) ? envs_[i]->reset(-1) : r.obs);
        }
    }

    // n x n_actions (column-major), empty when the environment has no legal mask.
    std::vector<uint8_t> legal_mask() const {
        std::vector<uint8_t> out;
        for (int i = 0; i < n(); i++) {
            std::vector<uint8_t> m = envs_[i]->legal_mask();
            if (m.empty()) return {};
            if (out.empty()) out.resize(m.size() * n());
            for (size_t j = 0; j < m.size(); j++) out[i + j * n()] = m[j];
        }
        return out;
    }

private:
    void put(double* dst, int i, const std::vector<float>& v) const {
        for (int j = 0; j < obs_dim_; j++) dst[i + j * n()] = v[j];
    }

    std::vector<std::unique_ptr<Env>> envs_;
    int obs_dim_ = 0;
    int act_dim_ = 0;
};

}  // namespace rltr
