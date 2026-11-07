// Continuous-control environments: pendulum, acrobot, car.
// Ported from RLtools (https://github.com/rl-tools/rl-tools,
// include/rl_tools/rl/environments/{pendulum,acrobot,car}); the acrobot dynamics
// follow Gymnasium v0.28.1 classic_control/acrobot.py. Physics in float, RNG is
// std::mt19937 seeded with seed + 1, as in RLtools.

#include <cmath>
#include <limits>
#include <random>

#include "car_track_data.h"
#include "env.h"

namespace rltr {

namespace {

using T = float;
using RNG = std::mt19937;

constexpr T PI = static_cast<T>(3.14159265358979323846);
constexpr double INF = std::numeric_limits<double>::infinity();

inline T clampf(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

// Floored modulo: wraps x into [-pi, pi).
inline T angle_normalize(T x) {
    T a = x + PI, b = 2 * PI;
    return a - b * std::floor(a / b) - PI;
}

template <typename R>
inline R uniform(R lo, R hi, RNG& rng) { return std::uniform_real_distribution<R>(lo, hi)(rng); }

// Wraps a model with a POD state. A model provides State, OBS_DIM, ACTION_DIM and
// sample_initial_state / step / reward / terminated / observe.
template <typename MODEL>
class ControlEnv : public Env {
public:
    using State = typename MODEL::State;

    ControlEnv(int64_t seed, EnvSpec spec) : spec_(std::move(spec)) {
        seed_rng(seed < 0 ? 0 : seed);
        model_.sample_initial_state(state_, rng_);
    }

    const EnvSpec& spec() const override { return spec_; }

protected:
    std::vector<float> do_reset(int64_t seed) override {
        if (seed >= 0) seed_rng(seed);
        model_.sample_initial_state(state_, rng_);
        return observe(state_);
    }

    StepResult do_step(const std::vector<double>& actions) override {
        if (actions.size() != MODEL::ACTION_DIM) {
            throw std::invalid_argument(spec_.name + ": action must have length " +
                                        std::to_string(MODEL::ACTION_DIM));
        }
        T a[MODEL::ACTION_DIM];
        for (int i = 0; i < MODEL::ACTION_DIM; i++) a[i] = static_cast<T>(actions[i]);

        State next;
        model_.step(state_, a, next);
        StepResult r;
        r.reward = model_.reward(state_, a, next);
        r.terminated = model_.terminated(next);
        state_ = next;
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
    void seed_rng(int64_t seed) { rng_.seed(static_cast<RNG::result_type>(seed + 1)); }

    std::vector<float> observe(const State& s) const {
        std::vector<float> obs(MODEL::OBS_DIM);
        model_.observe(s, obs.data());
        return obs;
    }

    MODEL model_;
    RNG rng_;
    State state_{};
    EnvSpec spec_;
};

// ---------------------------------------------------------------- pendulum

struct Pendulum {
    static constexpr int OBS_DIM = 3, ACTION_DIM = 1, STEP_LIMIT = 200;
    static constexpr T g = 10, max_speed = 8, max_torque = 2, dt = 0.05f, m = 1, l = 1;

    struct State { T theta, theta_dot; };

    void sample_initial_state(State& s, RNG& rng) const {
        s.theta = uniform<T>(-PI, PI, rng);
        s.theta_dot = uniform<T>(-1, 1, rng);
    }

    void step(const State& s, const T* a, State& next) const {
        T u = max_torque * clampf(a[0], -1, 1);
        T thdot = s.theta_dot + (3 * g / (2 * l) * std::sin(s.theta) + 3 / (m * l * l) * u) * dt;
        thdot = clampf(thdot, -max_speed, max_speed);
        next.theta = s.theta + thdot * dt;
        next.theta_dot = thdot;
    }

    // As in RLtools, the torque cost uses the unclipped action.
    T reward(const State& s, const T* a, const State&) const {
        T th = angle_normalize(s.theta);
        T u = max_torque * a[0];
        return -(th * th + T(0.1) * s.theta_dot * s.theta_dot + T(0.001) * u * u);
    }

    bool terminated(const State&) const { return false; }

    void observe(const State& s, float* o) const {
        o[0] = std::cos(s.theta);
        o[1] = std::sin(s.theta);
        o[2] = s.theta_dot;
    }
};

// ----------------------------------------------------------------- acrobot

// SWINGUP = false: classic task (start near the bottom, -1 per step, terminates
// when the tip rises above the bar). SWINGUP = true: random start, dense reward
// for keeping both links upright, no termination.
template <bool SWINGUP>
struct Acrobot {
    static constexpr int OBS_DIM = 6, ACTION_DIM = 1, STEP_LIMIT = 500;
    static constexpr T DT = 0.2f;
    static constexpr T L1 = 1, M1 = 1, M2 = 1, LC1 = 0.5f, LC2 = 0.5f, MOI = 1;
    static constexpr T MAX_VEL_1 = 4 * PI, MAX_VEL_2 = 9 * PI;
    static constexpr T MIN_TORQUE = -5, MAX_TORQUE = 5;

    struct State { T theta_1, theta_2, theta_1_dot, theta_2_dot; };

    void sample_initial_state(State& s, RNG& rng) const {
        if (SWINGUP) {
            s.theta_1 = uniform<T>(-PI, PI, rng);
            s.theta_2 = uniform<T>(-PI, PI, rng);
            s.theta_1_dot = 0;
            s.theta_2_dot = 0;
        } else {
            s.theta_1 = static_cast<T>(uniform<double>(-0.1, 0.1, rng));
            s.theta_2 = static_cast<T>(uniform<double>(-0.1, 0.1, rng));
            s.theta_1_dot = static_cast<T>(uniform<double>(-0.1, 0.1, rng));
            s.theta_2_dot = static_cast<T>(uniform<double>(-0.1, 0.1, rng));
        }
    }

    static void dsdt(const T y[4], T torque, T dy[4]) {
        const T g = 9.8f, I1 = MOI, I2 = MOI;
        T th1 = y[0], th2 = y[1], dth1 = y[2], dth2 = y[3];
        T d1 = M1 * LC1 * LC1 + M2 * (L1 * L1 + LC2 * LC2 + 2 * L1 * LC2 * std::cos(th2)) + I1 + I2;
        T d2 = M2 * (LC2 * LC2 + L1 * LC2 * std::cos(th2)) + I2;
        T phi2 = M2 * LC2 * g * std::cos(th1 + th2 - PI / 2);
        T phi1 = -M2 * L1 * LC2 * dth2 * dth2 * std::sin(th2)
                 - 2 * M2 * L1 * LC2 * dth2 * dth1 * std::sin(th2)
                 + (M1 * LC1 + M2 * L1) * g * std::cos(th1 - PI / 2) + phi2;
        T ddth2 = (torque + d2 / d1 * phi1 - M2 * L1 * LC2 * dth1 * dth1 * std::sin(th2) - phi2) /
                  (M2 * LC2 * LC2 + I2 - d2 * d2 / d1);
        T ddth1 = -(d2 * ddth2 + phi1) / d1;
        dy[0] = dth1;
        dy[1] = dth2;
        dy[2] = ddth1;
        dy[3] = ddth2;
    }

    void step(const State& s, const T* a, State& next) const {
        T torque = (clampf(a[0], -1, 1) + 1) / 2 * (MAX_TORQUE - MIN_TORQUE) + MIN_TORQUE;
        T y[4] = {s.theta_1, s.theta_2, s.theta_1_dot, s.theta_2_dot};
        T k1[4], k2[4], k3[4], k4[4], t[4];
        const T h = DT / 2;
        dsdt(y, torque, k1);
        for (int i = 0; i < 4; i++) t[i] = y[i] + h * k1[i];
        dsdt(t, torque, k2);
        for (int i = 0; i < 4; i++) t[i] = y[i] + h * k2[i];
        dsdt(t, torque, k3);
        for (int i = 0; i < 4; i++) t[i] = y[i] + DT * k3[i];
        dsdt(t, torque, k4);
        for (int i = 0; i < 4; i++) y[i] += DT / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        next.theta_1 = angle_normalize(y[0]);
        next.theta_2 = angle_normalize(y[1]);
        next.theta_1_dot = clampf(y[2], -MAX_VEL_1, MAX_VEL_1);
        next.theta_2_dot = clampf(y[3], -MAX_VEL_2, MAX_VEL_2);
    }

    T reward(const State&, const T*, const State& next) const {
        if (!SWINGUP) return -1;
        T a = std::abs(angle_normalize(next.theta_1 - PI));
        T b = std::abs(angle_normalize(next.theta_2));
        return (-a - b) / 100 * DT / T(0.02);
    }

    bool terminated(const State& s) const {
        if (SWINGUP) return false;
        return -std::cos(s.theta_1) - std::cos(s.theta_2 + s.theta_1) > 1;
    }

    void observe(const State& s, float* o) const {
        o[0] = std::cos(s.theta_1);
        o[1] = std::sin(s.theta_1);
        o[2] = std::cos(s.theta_2);
        o[3] = std::sin(s.theta_2);
        o[4] = s.theta_1_dot;
        o[5] = s.theta_2_dot;
    }
};

// --------------------------------------------------------------------- car

// Dynamic bicycle model with Pacejka tires (1:43 scale RC car), dt = 0.01.
// TRACK = false: stabilize at the origin, terminates when leaving [-1, 1]^2.
// TRACK = true: race on the default track, reward = forward speed, terminates
// off-track; obs adds 3 normalized lidar distances (-20, 0, +20 degrees).
template <bool TRACK>
struct Car {
    static constexpr int OBS_DIM = TRACK ? 9 : 6, ACTION_DIM = 2;
    static constexpr T g = 9.81f, m = 0.041f, I = 27.8e-6f, lf = 0.029f, lr = 0.033f;
    static constexpr T tf_B = 2.5790f, tf_C = 1.2000f, tf_D = 0.1920f;
    static constexpr T tr_B = 3.3852f, tr_C = 1.2691f, tr_D = 0.1737f;
    static constexpr T cm = 0.287f, cr0 = 0, cr2 = 0.00035f, vt = 0.01f, dt = 0.01f;

    static constexpr int N = CAR_TRACK_SIZE;
    static constexpr T TRACK_SCALE = 20 / 1000.0f;  // 2 cm per cell
    static constexpr T HALF = N * TRACK_SCALE / 2;

    struct State { T x, y, mu, vx, vy, omega; };

    // Track cell under a point; false outside the map.
    static bool on_track(T x, T y) {
        T cx = (x + HALF) / TRACK_SCALE;
        T cy = (-y + HALF) / TRACK_SCALE;
        if (!(cx > 0 && cx < N && cy > 0 && cy < N)) return false;
        return CAR_TRACK[static_cast<int>(cy)][static_cast<int>(cx)] == '#';
    }

    void sample_initial_state(State& s, RNG& rng) const {
        s = State{};
        if (TRACK) {
            do {
                s.x = uniform<T>(-HALF, HALF, rng);
                s.y = uniform<T>(-HALF, HALF, rng);
                s.mu = uniform<T>(-PI, PI, rng);
            } while (terminated(s));
        } else {
            const T dist = 0.2f;
            s.x = uniform<T>(-dist, dist, rng);
            s.y = uniform<T>(-dist, dist, rng);
            s.mu = uniform<T>(-PI, PI, rng);
        }
    }

    void step(const State& s, const T* a, State& next) const {
        T throttle = clampf(a[0], -1, 1);
        T delta = clampf(a[1], -1, 1);
        T alpha_f = std::atan2(s.vy + lf * s.omega, s.vx) - delta;
        T alpha_r = std::atan2(s.vy - lr * s.omega, s.vx);
        T FnF = lr / (lf + lr) * m * g;
        T FnR = lf / (lf + lr) * m * g;
        bool moving = s.vx > vt || s.vy > vt;
        T FyF = moving ? FnF * tf_D * std::sin(tf_C * std::atan(tf_B * -alpha_f)) : 0;
        T FyR = moving ? FnR * tr_D * std::sin(tr_C * std::atan(tr_B * -alpha_r)) : 0;
        T Fx = cm * throttle - cr0 - cr2 * s.vx * s.vx;
        T c = std::cos(s.mu), sn = std::sin(s.mu);
        next = s;
        next.x += dt * (c * s.vx - sn * s.vy);
        next.y += dt * (sn * s.vx + c * s.vy);
        next.mu += dt * s.omega;
        next.vx += dt * (1 / m * (Fx - FyF * std::sin(delta)) + s.omega * s.vy);
        next.vy += dt * (1 / m * (FyF * std::cos(delta) + FyR) - s.omega * s.vx);
        next.omega += dt * (1 / I * (FyF * lf * std::cos(delta) - FyR * lr));
    }

    T reward(const State& s, const T*, const State&) const {
        if (TRACK) return s.vx;
        T th = angle_normalize(s.mu);
        T cost = T(0.1) * th * th + s.x * s.x + s.y * s.y;
        return std::exp(-5 * cost);
    }

    bool terminated(const State& s) const {
        if (TRACK) return !on_track(s.x, s.y);
        return s.x > 1 || s.x < -1 || s.y > 1 || s.y < -1;
    }

    void observe(const State& s, float* o) const {
        o[0] = s.x;
        o[1] = s.y;
        o[2] = s.mu;
        o[3] = s.vx;
        o[4] = s.vy;
        o[5] = s.omega;
        if (!TRACK) return;
        // Ray march; RLtools indexed the map without bounds checks, here leaving
        // the map counts as hitting a wall.
        const int NUM_STEPS = 50;
        const T step_size = TRACK_SCALE / 2;
        const T dirs[3] = {-20 / 180.0f * PI, 0, 20 / 180.0f * PI};
        for (int d = 0; d < 3; d++) {
            T dx = std::cos(s.mu + dirs[d]) * step_size;
            T dy = std::sin(s.mu + dirs[d]) * step_size;
            int hit = NUM_STEPS;
            for (int i = 0; i < NUM_STEPS; i++) {
                if (!on_track(s.x + dx * i, s.y + dy * i)) { hit = i; break; }
            }
            o[6 + d] = static_cast<T>(hit) / NUM_STEPS;
        }
    }
};

// RLtools defines no step limit for the car; 1000 steps = 10 s at dt = 0.01.
constexpr int CAR_MAX_STEPS = 1000;

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

template <bool SWINGUP>
Env* make_acrobot_impl(const std::string& name, int64_t seed) {
    using M = Acrobot<SWINGUP>;
    return new ControlEnv<M>(seed, make_spec(
        name, M::STEP_LIMIT,
        box({-1, -1, -1, -1, -M::MAX_VEL_1, -M::MAX_VEL_2}, {1, 1, 1, 1, M::MAX_VEL_1, M::MAX_VEL_2}),
        box({-1}, {1})));
}

}  // namespace

Env* make_pendulum(int64_t seed) {
    return new ControlEnv<Pendulum>(seed, make_spec(
        "pendulum", Pendulum::STEP_LIMIT,
        box({-1, -1, -Pendulum::max_speed}, {1, 1, Pendulum::max_speed}),
        box({-1}, {1})));
}

Env* make_acrobot(int64_t seed) { return make_acrobot_impl<false>("acrobot", seed); }

Env* make_acrobot_swingup(int64_t seed) { return make_acrobot_impl<true>("acrobot_swingup", seed); }

Env* make_car(int64_t seed) {
    return new ControlEnv<Car<false>>(seed, make_spec(
        "car", CAR_MAX_STEPS,
        box({-INF, -INF, -INF, -INF, -INF, -INF}, {INF, INF, INF, INF, INF, INF}),
        box({-1, -1}, {1, 1})));
}

Env* make_car_track(int64_t seed) {
    return new ControlEnv<Car<true>>(seed, make_spec(
        "car_track", CAR_MAX_STEPS,
        box({-INF, -INF, -INF, -INF, -INF, -INF, 0, 0, 0}, {INF, INF, INF, INF, INF, INF, 1, 1, 1}),
        box({-1, -1}, {1, 1})));
}

}  // namespace rltr
