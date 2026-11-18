// Tabular environments given by a transition table, as Gymnasium's toy_text
// DiscreteEnv: P[s][a] is a list of K outcomes (prob, next state, reward,
// terminated). Lists shorter than K are padded with probability 0. The outcome
// is the first k with cumsum(prob)[k] > u, u ~ U[0, 1), as Gymnasium's
// categorical_sample, so padding is never chosen.
#pragma once

#include <random>

#include "env.h"

namespace rltr {

struct Outcome {
    double prob = 0;
    int next = 0;
    double reward = 0;
    bool terminated = false;
};

struct TabularTable {
    int S = 0, A = 0, K = 0;
    std::vector<Outcome> P;     // index (s * A + a) * K + k
    std::vector<double> isd;    // initial state distribution, length S
    // Optional observation table: row s (obs_dim values, row-major S x obs_dim)
    // is the observation of state s, in obs_space. Empty: the state index or
    // its one-hot.
    int obs_dim = 0;
    std::vector<float> obs;
    Space obs_space;

    Outcome& at(int s, int a, int k) { return P[(static_cast<size_t>(s) * A + a) * K + k]; }
    const Outcome& at(int s, int a, int k) const { return P[(static_cast<size_t>(s) * A + a) * K + k]; }
};

// Observation: the row of the table's observation table when it has one,
// else the state index (Discrete(S)) or, with onehot, a one-hot vector of
// length S (Box[0, 1]).
class TabularEnv : public Env {
public:
    TabularEnv(int64_t seed, std::string name, int max_steps, TabularTable table, bool onehot)
        : table_(std::move(table)), onehot_(onehot) {
        spec_.name = std::move(name);
        spec_.max_steps = max_steps;
        Space& o = spec_.observation_space;
        if (table_.obs_dim > 0) {
            o = table_.obs_space;
        } else if (onehot) {
            o.type = SpaceType::Box;
            o.shape = {table_.S};
            o.low.assign(table_.S, 0);
            o.high.assign(table_.S, 1);
        } else {
            o.type = SpaceType::Discrete;
            o.dtype = DType::Int32;
            o.n = table_.S;
        }
        Space& a = spec_.action_space;
        a.type = SpaceType::Discrete;
        a.dtype = DType::Int32;
        a.n = table_.A;
        seed_rng(seed < 0 ? 0 : seed);
        state_ = sample_initial();
    }

    const EnvSpec& spec() const override { return spec_; }
    const TabularTable& table() const { return table_; }

protected:
    std::vector<float> do_reset(int64_t seed) override {
        if (seed >= 0) seed_rng(seed);
        state_ = sample_initial();
        return observe();
    }

    StepResult do_step(const std::vector<double>& actions) override {
        if (actions.size() != 1) throw std::invalid_argument(spec_.name + ": action must have length 1");
        const double a = actions[0];
        if (!(a >= 0 && a < table_.A) || a != static_cast<int>(a)) {
            throw std::invalid_argument(spec_.name + ": action must be an integer in 0.." +
                                        std::to_string(table_.A - 1));
        }
        const double u = unif_(rng_);
        double cum = 0;
        const Outcome* o = nullptr;
        const Outcome* last = nullptr;   // the last outcome with probability > 0
        for (int k = 0; k < table_.K; k++) {
            const Outcome& c = table_.at(state_, static_cast<int>(a), k);
            if (c.prob > 0) last = &c;
            cum += c.prob;
            if (cum > u) { o = &c; break; }
        }
        // rounding: a sum a little below 1 falls back to the last outcome
        if (o == nullptr && last != nullptr && cum > 1 - 1e-6) o = last;
        if (o == nullptr) throw std::runtime_error(spec_.name + ": transition probabilities sum below 1");
        state_ = o->next;
        StepResult r;
        r.reward = o->reward;
        r.terminated = o->terminated;
        r.obs = observe();
        return r;
    }

    std::vector<uint8_t> do_save_state() const override {
        std::vector<uint8_t> out(sizeof(state_));
        std::memcpy(out.data(), &state_, sizeof(state_));
        return out;
    }

    void do_load_state(const std::vector<uint8_t>& payload) override {
        int s;
        if (payload.size() != sizeof(s)) throw std::runtime_error(spec_.name + ": invalid state snapshot");
        std::memcpy(&s, payload.data(), sizeof(s));
        if (s < 0 || s >= table_.S) throw std::runtime_error(spec_.name + ": state out of range");
        state_ = s;
    }

private:
    void seed_rng(int64_t seed) { rng_.seed(static_cast<std::mt19937::result_type>(seed + 1)); }

    int sample_initial() {
        const double u = unif_(rng_);
        double cum = 0;
        for (int s = 0; s < table_.S; s++) {
            cum += table_.isd[s];
            if (cum > u) return s;
        }
        for (int s = table_.S - 1; s >= 0; s--) if (table_.isd[s] > 0) return s;
        throw std::runtime_error(spec_.name + ": empty initial state distribution");
    }

    std::vector<float> observe() const {
        if (table_.obs_dim > 0) {
            const float* row = table_.obs.data() + static_cast<size_t>(state_) * table_.obs_dim;
            return std::vector<float>(row, row + table_.obs_dim);
        }
        if (!onehot_) return {static_cast<float>(state_)};
        std::vector<float> o(table_.S, 0.0f);
        o[state_] = 1.0f;
        return o;
    }

    TabularTable table_;
    bool onehot_;
    EnvSpec spec_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> unif_{0.0, 1.0};
    int state_ = 0;
};

}  // namespace rltr
