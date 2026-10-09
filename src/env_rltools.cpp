// Continuous-control environments from RLtools: pendulum, acrobot, car.
#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_cpu.h>
// operations_cpu.h only adds a broken BMP track loader that prints to std::cout
#include <rl_tools/rl/environments/car/operations_generic.h>

#include <limits>

#include "env.h"

namespace rlt = rl_tools;

namespace rltr {

namespace {

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = DEVICE::index_t;
using RNG = rlt::devices::random::CPU::ENGINE<>;

constexpr double INF = std::numeric_limits<double>::infinity();

// Wraps any RLtools environment with a POD state and a single-row observation.
template <typename ENVIRONMENT>
class RltEnv : public Env {
public:
    static constexpr TI OBS_DIM = ENVIRONMENT::Observation::DIM;
    static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;

    RltEnv(int64_t seed, EnvSpec spec) : spec_(std::move(spec)) {
        rlt::init(device_, rng_, seed < 0 ? 0 : static_cast<TI>(seed));
        rlt::init(device_, env_);
        rlt::initial_parameters(device_, env_, params_);
        rlt::sample_initial_state(device_, env_, params_, state_, rng_);
    }

    const EnvSpec& spec() const override { return spec_; }

protected:
    std::vector<float> do_reset(int64_t seed) override {
        if (seed >= 0) rlt::init(device_, rng_, static_cast<TI>(seed));
        rlt::sample_initial_state(device_, env_, params_, state_, rng_);
        return observe(state_);
    }

    StepResult do_step(const std::vector<double>& actions) override {
        if (actions.size() != ACTION_DIM) {
            throw std::invalid_argument(spec_.name + ": action must have length " + std::to_string(ACTION_DIM));
        }
        for (TI i = 0; i < ACTION_DIM; i++) rlt::set(action_, 0, i, static_cast<T>(actions[i]));

        typename ENVIRONMENT::State next_state;
        rlt::step(device_, env_, params_, state_, action_, next_state, rng_);

        StepResult r;
        r.reward = rlt::reward(device_, env_, params_, state_, action_, next_state, rng_);
        r.terminated = rlt::terminated(device_, env_, params_, next_state, rng_);
        state_ = next_state;
        r.obs = observe(state_);
        return r;
    }

    std::vector<uint8_t> do_save_state() const override {
        std::vector<uint8_t> out(sizeof(state_));
        std::memcpy(out.data(), &state_, sizeof(state_));
        return out;
    }

    void do_load_state(const std::vector<uint8_t>& payload) override {
        if (payload.size() != sizeof(state_)) throw std::runtime_error(spec_.name + ": invalid state snapshot");
        std::memcpy(&state_, payload.data(), sizeof(state_));
    }

private:
    std::vector<float> observe(const typename ENVIRONMENT::State& state) {
        rlt::observe(device_, env_, params_, state, typename ENVIRONMENT::Observation{}, observation_, rng_);
        std::vector<float> obs(OBS_DIM);
        for (TI i = 0; i < OBS_DIM; i++) obs[i] = rlt::get(observation_, 0, i);
        return obs;
    }

    DEVICE device_;
    RNG rng_;
    ENVIRONMENT env_;
    typename ENVIRONMENT::Parameters params_;
    typename ENVIRONMENT::State state_;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM, false>> action_;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS_DIM, false>> observation_;
    EnvSpec spec_;
};

Space box(std::vector<double> low, std::vector<double> high) {
    Space s;
    s.type = SpaceType::Box;
    s.shape = {static_cast<int>(low.size())};
    s.low = std::move(low);
    s.high = std::move(high);
    return s;
}

EnvSpec make_spec(const std::string& name, int max_steps, Space obs, Space act) {
    EnvSpec s;
    s.name = name;
    s.max_steps = max_steps;
    s.observation_space = std::move(obs);
    s.action_space = std::move(act);
    return s;
}

}  // namespace

Env* make_pendulum(int64_t seed) {
    using PARAMS = rlt::rl::environments::pendulum::DefaultParameters<T>;
    using ENVIRONMENT = rlt::rl::environments::Pendulum<rlt::rl::environments::pendulum::Specification<T, TI, PARAMS>>;
    return new RltEnv<ENVIRONMENT>(seed, make_spec(
        "pendulum", ENVIRONMENT::EPISODE_STEP_LIMIT,
        box({-1, -1, -PARAMS::max_speed}, {1, 1, PARAMS::max_speed}),
        box({-1}, {1})));
}

template <template <typename> class ACROBOT>
static Env* make_acrobot_impl(const std::string& name, int64_t seed) {
    using PARAMS = rlt::rl::environments::acrobot::DefaultParameters<T>;
    using ENVIRONMENT = ACROBOT<rlt::rl::environments::acrobot::Specification<T, TI, PARAMS>>;
    return new RltEnv<ENVIRONMENT>(seed, make_spec(
        name, ENVIRONMENT::EPISODE_STEP_LIMIT,
        box({-1, -1, -1, -1, -PARAMS::MAX_VEL_1, -PARAMS::MAX_VEL_2},
            {1, 1, 1, 1, PARAMS::MAX_VEL_1, PARAMS::MAX_VEL_2}),
        box({-1}, {1})));
}

// Classic task: reward -1 per step, terminates when the tip rises above the bar.
Env* make_acrobot(int64_t seed) {
    return make_acrobot_impl<rlt::rl::environments::Acrobot>("acrobot", seed);
}

// Swing-up: dense reward for keeping both links upright, no termination.
Env* make_acrobot_swingup(int64_t seed) {
    return make_acrobot_impl<rlt::rl::environments::AcrobotSwingup>("acrobot_swingup", seed);
}

// RLtools defines no step limit for the car; 1000 steps = 10 s at dt = 0.01.
constexpr int CAR_MAX_STEPS = 1000;

// Stabilize at the origin; terminates when leaving [-1, 1]^2.
Env* make_car(int64_t seed) {
    using ENVIRONMENT = rlt::rl::environments::Car<rlt::rl::environments::car::Specification<T, TI>>;
    return new RltEnv<ENVIRONMENT>(seed, make_spec(
        "car", CAR_MAX_STEPS,
        box({-INF, -INF, -INF, -INF, -INF, -INF}, {INF, INF, INF, INF, INF, INF}),
        box({-1, -1}, {1, 1})));
}

// Race on the default 100x100 track; obs adds 3 normalized lidar distances.
Env* make_car_track(int64_t seed) {
    using ENVIRONMENT = rlt::rl::environments::CarTrack<rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 20>>;
    return new RltEnv<ENVIRONMENT>(seed, make_spec(
        "car_track", CAR_MAX_STEPS,
        box({-INF, -INF, -INF, -INF, -INF, -INF, 0, 0, 0}, {INF, INF, INF, INF, INF, INF, 1, 1, 1}),
        box({-1, -1}, {1, 1})));
}

}  // namespace rltr
