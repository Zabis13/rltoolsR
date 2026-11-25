// Box2D environments (Gymnasium box2d) on the bundled Box2D v3 (src/box2d).
//
// LunarLander-v3 (gymnasium/envs/box2d/lunar_lander.py, enable_wind = False):
// constants, terrain, lander and legs, engines, observation, reward shaping
// and episode ends as in the Python source. Differences, all from Box2D v3 or
// our RNG:
//   - the solver: 8 sub-steps per step and leg joints at 120 Hz (Box2D 2.3
//     used 180 velocity and 60 position iterations and rigid joints); see
//     LL_SUBSTEPS below;
//   - the leg contact flags are read from the legs' touching contacts each
//     step (Gymnasium counts begin / end events, which a leg touching two
//     terrain segments would clear too early in v3);
//   - "the lander fell asleep" (+100) uses v3's sleep thresholds, so it may
//     come a few steps earlier or later;
//   - terrain, the initial push and the engine dispersion come from our RNG
//     (std::mt19937 seeded with seed + 1), not numpy's;
//   - the exhaust particles are left out: they collide with the ground only,
//     never with the lander, so they do not change its flight;
//   - Box2D v3 polygons have no skin radius (2.3: 0.01 m).
// The angle in the observation is unwrapped (continuous over turns), as
// Box2D 2.3's body angle.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <random>
#include <stdexcept>

#include "box2d/box2d.h"
#include "env.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

namespace rltr {

namespace {

constexpr double LL_FPS = 50, LL_SCALE = 30;
constexpr double MAIN_ENGINE_POWER = 13, SIDE_ENGINE_POWER = 0.6;
constexpr double INITIAL_RANDOM = 1000;
constexpr double LEG_AWAY = 20, LEG_DOWN = 18, LEG_W = 2, LEG_H = 8, LEG_SPRING_TORQUE = 40;
constexpr double SIDE_ENGINE_HEIGHT = 14, SIDE_ENGINE_AWAY = 12, MAIN_ENGINE_Y_LOCATION = 4;
constexpr double VIEWPORT_W = 600, VIEWPORT_H = 400;
constexpr int LL_CHUNKS = 11, LL_MAX_STEPS = 1000;
// Solver settings, chosen with Gymnasium's heuristic as the yardstick (in
// Gymnasium it lands almost always, mean return above 200). Grid on seeds
// 1-20 (sub-steps 4 / 8 / 16 x joint hertz 60 / 120 / 240 x contact hertz
// 30 / 60 x sleep threshold 0.05 / 0.1): 4 sub-steps 17 / 20 landed with an
// episode at the step limit, 8 or 16 sub-steps with 120 Hz joints 19 / 20,
// contact hertz 60 worse, sleep threshold no effect. Checked on 200 new seeds
// (1001-1200): 8 sub-steps 92.0% landed, mean return 240.8; 16 sub-steps
// 93.0%, 245.7; 12 of the 16 / 14 failures are the same seeds (the
// heuristic's, not the physics'). 8 sub-steps: half the cost of 16.
constexpr int LL_SUBSTEPS = 8;
constexpr float LL_JOINT_HERTZ = 120.0f;   // Box2D v3 default 60; damping ratio 2 (default)
constexpr double LL_PI = 3.14159265358979323846;
const double LANDER_POLY[6][2] = {{-14, 17}, {-17, 0}, {-17, -10}, {17, -10}, {17, 0}, {14, 17}};

constexpr uint64_t CAT_GROUND = 0x0001, CAT_LANDER = 0x0010, CAT_LEG = 0x0020;

// The lander's body: the hull of LANDER_POLY at (x0, y0), colliding with the
// ground only. Shared by LunarLander::build() and rlt_box2d_check().
b2BodyId make_hull(b2WorldId world, double x0, double y0) {
    b2BodyDef ld = b2DefaultBodyDef();
    ld.type = b2_dynamicBody;
    ld.position = {static_cast<float>(x0), static_cast<float>(y0)};
    b2BodyId lander = b2CreateBody(world, &ld);
    b2Vec2 pts[6];
    for (int i = 0; i < 6; i++)
        pts[i] = {static_cast<float>(LANDER_POLY[i][0] / LL_SCALE), static_cast<float>(LANDER_POLY[i][1] / LL_SCALE)};
    b2Hull hull = b2ComputeHull(pts, 6);
    b2Polygon body_poly = b2MakePolygon(&hull, 0.0f);
    b2ShapeDef ls = b2DefaultShapeDef();
    ls.density = 5.0f;
    ls.material.friction = 0.1f;
    ls.material.restitution = 0.0f;
    ls.filter.categoryBits = CAT_LANDER;
    ls.filter.maskBits = CAT_GROUND;
    ls.enableContactEvents = true;
    b2CreatePolygonShape(lander, &ls, &body_poly);
    return lander;
}

// The two legs on revolute joints with spring motors, as in Gymnasium.
void make_legs(b2WorldId world, b2BodyId lander, double x0, double y0, b2BodyId legs[2]) {
    for (int k = 0; k < 2; k++) {
        const int i = k == 0 ? -1 : 1;
        b2BodyDef gd = b2DefaultBodyDef();
        gd.type = b2_dynamicBody;
        gd.position = {static_cast<float>(x0 - i * LEG_AWAY / LL_SCALE), static_cast<float>(y0)};
        gd.rotation = b2MakeRot(static_cast<float>(i * 0.05));
        legs[k] = b2CreateBody(world, &gd);
        b2Polygon leg = b2MakeBox(static_cast<float>(LEG_W / LL_SCALE), static_cast<float>(LEG_H / LL_SCALE));
        b2ShapeDef s = b2DefaultShapeDef();
        s.density = 1.0f;
        s.material.friction = 0.2f;
        s.material.restitution = 0.0f;
        s.filter.categoryBits = CAT_LEG;
        s.filter.maskBits = CAT_GROUND;
        s.enableContactEvents = true;
        b2CreatePolygonShape(legs[k], &s, &leg);

        b2RevoluteJointDef jd = b2DefaultRevoluteJointDef();
        jd.base.bodyIdA = lander;
        jd.base.bodyIdB = legs[k];
        jd.base.localFrameA.p = {0.0f, 0.0f};
        jd.base.localFrameB.p = {static_cast<float>(i * LEG_AWAY / LL_SCALE), static_cast<float>(LEG_DOWN / LL_SCALE)};
        jd.base.constraintHertz = LL_JOINT_HERTZ;
        jd.enableMotor = true;
        jd.enableLimit = true;
        jd.maxMotorTorque = static_cast<float>(LEG_SPRING_TORQUE);
        jd.motorSpeed = static_cast<float>(0.3 * i);
        if (i == -1) {
            jd.lowerAngle = 0.9f - 0.5f;
            jd.upperAngle = 0.9f;
        } else {
            jd.lowerAngle = -0.9f;
            jd.upperAngle = -0.9f + 0.5f;
        }
        b2CreateRevoluteJoint(world, &jd);
    }
}

// Everything besides the Box2D world that a snapshot must restore (POD).
struct LunarState {
    b2BodyId moon, lander, legs[2];
    double helipad_y;
    double prev_shaping;
    double angle;          // unwrapped lander angle
    bool has_prev, game_over, contact[2];
};

class LunarLander : public Env {
public:
    LunarLander(int64_t seed, bool continuous) : continuous_(continuous) {
        spec_.name = continuous ? "lunar_lander_continuous" : "lunar_lander";
        spec_.max_steps = LL_MAX_STEPS;
        Space& o = spec_.observation_space;
        o.type = SpaceType::Box;
        o.shape = {8};
        o.low = {-2.5, -2.5, -10, -10, -2 * LL_PI, -10, 0, 0};
        o.high = {2.5, 2.5, 10, 10, 2 * LL_PI, 10, 1, 1};
        Space& a = spec_.action_space;
        if (continuous) {
            a.type = SpaceType::Box;
            a.shape = {2};
            a.low = {-1, -1};
            a.high = {1, 1};
        } else {
            a.type = SpaceType::Discrete;
            a.dtype = DType::Int32;
            a.n = 4;
        }
        b2WorldDef wd = b2DefaultWorldDef();
        wd.gravity = {0.0f, -10.0f};
        wd.workerCount = 1;
        world_ = b2CreateWorld(&wd);
        rng_.seed(static_cast<std::mt19937::result_type>((seed < 0 ? 0 : seed) + 1));
        build();
    }

    ~LunarLander() override { b2DestroyWorld(world_); }

    const EnvSpec& spec() const override { return spec_; }

protected:
    std::vector<float> do_reset(int64_t seed) override {
        if (seed >= 0) rng_.seed(static_cast<std::mt19937::result_type>(seed + 1));
        build();
        // Gymnasium's reset() returns the observation of one no-op step
        std::vector<float> obs;
        double reward;
        bool terminated;
        physics_step(0.0, 0.0, 0, obs, reward, terminated);
        return obs;
    }

    StepResult do_step(const std::vector<double>& actions) override {
        double main = 0, lateral = 0;
        int discrete = 0;
        if (continuous_) {
            if (actions.size() != 2) throw std::invalid_argument(spec_.name + ": action must have length 2");
            if (std::isnan(actions[0]) || std::isnan(actions[1]))
                throw std::invalid_argument(spec_.name + ": action must not be NaN");
            main = std::min(std::max(actions[0], -1.0), 1.0);
            lateral = std::min(std::max(actions[1], -1.0), 1.0);
        } else {
            if (actions.size() != 1) throw std::invalid_argument(spec_.name + ": action must have length 1");
            const double x = actions[0];
            if (!(x >= 0 && x <= 3) || x != static_cast<int>(x))
                throw std::invalid_argument(spec_.name + ": action must be an integer in 0..3");
            discrete = static_cast<int>(x);
        }
        StepResult r;
        physics_step(main, lateral, discrete, r.obs, r.reward, r.terminated);
        return r;
    }

    std::vector<uint8_t> do_save_state() const override {
        const int n = b2World_GetSnapshot(world_, nullptr, 0);
        if (n <= 0) throw std::runtime_error(spec_.name + ": Box2D snapshot failed");
        std::vector<uint8_t> out(sizeof(LunarState) + static_cast<size_t>(n));
        std::memcpy(out.data(), &st_, sizeof(LunarState));
        b2World_GetSnapshot(world_, out.data() + sizeof(LunarState), n);
        return out;
    }

    void do_load_state(const std::vector<uint8_t>& payload) override {
        if (payload.size() <= sizeof(LunarState)) throw std::runtime_error(spec_.name + ": invalid state snapshot");
        const int n = static_cast<int>(payload.size() - sizeof(LunarState));
        if (!b2World_RestoreFromSnapshot(world_, payload.data() + sizeof(LunarState), n))
            throw std::runtime_error(spec_.name + ": Box2D could not restore the snapshot");
        std::memcpy(&st_, payload.data(), sizeof(LunarState));
    }

private:
    double unif(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng_); }

    // A new terrain, lander and legs (Gymnasium's reset() up to its first step).
    void build() {
        for (b2BodyId b : {st_.moon, st_.lander, st_.legs[0], st_.legs[1]})
            if (b2Body_IsValid(b)) b2DestroyBody(b);
        st_ = LunarState{};

        const double W = VIEWPORT_W / LL_SCALE, H = VIEWPORT_H / LL_SCALE;
        double height[LL_CHUNKS + 1];
        for (double& h : height) h = unif(0, H / 2);
        double chunk_x[LL_CHUNKS];
        for (int i = 0; i < LL_CHUNKS; i++) chunk_x[i] = W / (LL_CHUNKS - 1) * i;
        st_.helipad_y = H / 4;
        for (int k = -2; k <= 2; k++) height[LL_CHUNKS / 2 + k] = st_.helipad_y;
        double smooth_y[LL_CHUNKS];
        for (int i = 0; i < LL_CHUNKS; i++) {
            // height[i - 1] at i = 0 is Python's height[-1], the last element
            const double prev = height[i == 0 ? LL_CHUNKS : i - 1];
            smooth_y[i] = 0.33 * (prev + height[i] + height[i + 1]);
        }

        b2BodyDef md = b2DefaultBodyDef();
        st_.moon = b2CreateBody(world_, &md);
        b2ShapeDef gs = b2DefaultShapeDef();
        gs.density = 0;
        gs.material.friction = 0.2f;      // Box2D 2.3's default fixture friction
        gs.filter.categoryBits = CAT_GROUND;
        gs.enableContactEvents = true;
        b2Segment base = {{0.0f, 0.0f}, {static_cast<float>(W), 0.0f}};
        b2CreateSegmentShape(st_.moon, &gs, &base);
        gs.material.friction = 0.1f;
        for (int i = 0; i < LL_CHUNKS - 1; i++) {
            b2Segment seg = {{static_cast<float>(chunk_x[i]), static_cast<float>(smooth_y[i])},
                             {static_cast<float>(chunk_x[i + 1]), static_cast<float>(smooth_y[i + 1])}};
            b2CreateSegmentShape(st_.moon, &gs, &seg);
        }

        const double x0 = W / 2, y0 = H;
        st_.lander = make_hull(world_, x0, y0);
        const float fx = static_cast<float>(unif(-INITIAL_RANDOM, INITIAL_RANDOM));
        const float fy = static_cast<float>(unif(-INITIAL_RANDOM, INITIAL_RANDOM));
        b2Body_ApplyForceToCenter(st_.lander, {fx, fy}, true);
        make_legs(world_, st_.lander, x0, y0, st_.legs);
    }

    bool is(b2BodyId a, b2BodyId b) const { return B2_ID_EQUALS(a, b); }

    // Gymnasium's step(): engines, one world step, contacts, observation, reward.
    void physics_step(double main, double lateral, int discrete,
                      std::vector<float>& obs, double& reward, bool& terminated) {
        const b2BodyId L = st_.lander;
        const double ang = b2Rot_GetAngle(b2Body_GetRotation(L));
        const double tip[2] = {std::sin(ang), std::cos(ang)};
        const double side[2] = {-tip[1], tip[0]};
        const double disp[2] = {unif(-1, 1) / LL_SCALE, unif(-1, 1) / LL_SCALE};

        double m_power = 0;
        if ((continuous_ && main > 0) || (!continuous_ && discrete == 2)) {
            m_power = continuous_ ? (std::min(std::max(main, 0.0), 1.0) + 1.0) * 0.5 : 1.0;
            const double off = MAIN_ENGINE_Y_LOCATION / LL_SCALE + 2 * disp[0];
            const double ox = tip[0] * off + side[0] * disp[1];
            const double oy = -tip[1] * off - side[1] * disp[1];
            const b2Vec2 p = b2Body_GetPosition(L);
            b2Body_ApplyLinearImpulse(
                L, {static_cast<float>(-ox * MAIN_ENGINE_POWER * m_power), static_cast<float>(-oy * MAIN_ENGINE_POWER * m_power)},
                {static_cast<float>(p.x + ox), static_cast<float>(p.y + oy)}, true);
        }
        double s_power = 0;
        if ((continuous_ && std::fabs(lateral) > 0.5) || (!continuous_ && (discrete == 1 || discrete == 3))) {
            double direction;
            if (continuous_) {
                direction = lateral > 0 ? 1 : -1;
                s_power = std::min(std::max(std::fabs(lateral), 0.5), 1.0);
            } else {
                direction = discrete - 2;
                s_power = 1.0;
            }
            const double off = 3 * disp[1] + direction * SIDE_ENGINE_AWAY / LL_SCALE;
            const double ox = tip[0] * disp[0] + side[0] * off;
            const double oy = -tip[1] * disp[0] - side[1] * off;
            const b2Vec2 p = b2Body_GetPosition(L);
            b2Body_ApplyLinearImpulse(
                L, {static_cast<float>(-ox * SIDE_ENGINE_POWER * s_power), static_cast<float>(-oy * SIDE_ENGINE_POWER * s_power)},
                {static_cast<float>(p.x + ox - tip[0] * 17 / LL_SCALE),
                 static_cast<float>(p.y + oy + tip[1] * SIDE_ENGINE_HEIGHT / LL_SCALE)},
                true);
        }

        b2World_Step(world_, static_cast<float>(1.0 / LL_FPS), LL_SUBSTEPS);
        read_contacts();

        // unwrapped angle: add the step's turn, wrapped to (-pi, pi]
        const double ang1 = b2Rot_GetAngle(b2Body_GetRotation(L));
        double d = ang1 - ang;
        if (d > LL_PI) d -= 2 * LL_PI;
        if (d < -LL_PI) d += 2 * LL_PI;
        st_.angle += d;

        const b2Vec2 pos = b2Body_GetPosition(L);
        const b2Vec2 vel = b2Body_GetLinearVelocity(L);
        const double hw = VIEWPORT_W / LL_SCALE / 2, hh = VIEWPORT_H / LL_SCALE / 2;
        const double s[8] = {(pos.x - hw) / hw,
                             (pos.y - (st_.helipad_y + LEG_DOWN / LL_SCALE)) / hh,
                             vel.x * hw / LL_FPS,
                             vel.y * hh / LL_FPS,
                             st_.angle,
                             20.0 * b2Body_GetAngularVelocity(L) / LL_FPS,
                             st_.contact[0] ? 1.0 : 0.0,
                             st_.contact[1] ? 1.0 : 0.0};
        obs.assign(s, s + 8);

        const double shaping = -100 * std::sqrt(s[0] * s[0] + s[1] * s[1]) -
                               100 * std::sqrt(s[2] * s[2] + s[3] * s[3]) - 100 * std::fabs(s[4]) +
                               10 * s[6] + 10 * s[7];
        reward = st_.has_prev ? shaping - st_.prev_shaping : 0;
        st_.prev_shaping = shaping;
        st_.has_prev = true;
        reward -= m_power * 0.30;
        reward -= s_power * 0.03;

        terminated = false;
        if (st_.game_over || std::fabs(s[0]) >= 1.0) {
            terminated = true;
            reward = -100;
        }
        if (!b2Body_IsAwake(L)) {
            terminated = true;
            reward = 100;
        }
    }

    // Gymnasium's ContactDetector: the lander touching the ground ends the
    // game (a begin event). A leg's flag is whether it touches the ground now:
    // read from the leg's touching contacts every step, not counted from
    // begin / end events -- a leg on the joint of two terrain segments has
    // two contacts, and the end of one must not clear the flag.
    void read_contacts() {
        const b2ContactEvents ev = b2World_GetContactEvents(world_);
        for (int i = 0; i < ev.beginCount; i++) {
            const b2BodyId a = b2Shape_GetBody(ev.beginEvents[i].shapeIdA);
            const b2BodyId b = b2Shape_GetBody(ev.beginEvents[i].shapeIdB);
            if (is(a, st_.lander) || is(b, st_.lander)) st_.game_over = true;
        }
        b2ContactData data[8];
        for (int k = 0; k < 2; k++) {
            // legs collide with the ground only: any touching contact is ground
            st_.contact[k] = b2Body_GetContactData(st_.legs[k], data, 8) > 0;
        }
    }

    bool continuous_;
    EnvSpec spec_;
    b2WorldId world_;
    std::mt19937 rng_;
    LunarState st_{};
};

}  // namespace

Env* make_box2d(const std::string& name, int64_t seed) {
    if (name == "lunar_lander") return new LunarLander(seed, false);
    if (name == "lunar_lander_continuous") return new LunarLander(seed, true);
    return nullptr;
}

}  // namespace rltr

// Internal Box2D functions (src/box2d/src), declared here only to take their
// addresses for rlt_box2d_symbols(); never called through these declarations.
extern "C" {
void b2ComputeFatShapeAABB(void);
void b2GetBodyTransformQuick(void);
void b2UpdateBodyMassData(void);
}

// The shared object each Box2D function used on the segment's AABB path resolves
// to, as the dynamic linker binds it for this library (dladdr()). All should be
// this package's own library (the first entry, rlt_box2d_check itself); another
// file means a symbol of another library in the process overrides Box2D's
// (ELF symbol interposition, Linux). Empty strings on Windows.
extern "C" SEXP rlt_box2d_check(void);
extern "C" SEXP rlt_box2d_symbols(void) {
    struct Sym { const char* name; const void* addr; };
    const Sym syms[] = {
        {"rlt_box2d_check", reinterpret_cast<const void*>(&rlt_box2d_check)},
        {"b2ComputeFatShapeAABB", reinterpret_cast<const void*>(&b2ComputeFatShapeAABB)},
        {"b2GetBodyTransformQuick", reinterpret_cast<const void*>(&b2GetBodyTransformQuick)},
        {"b2ComputeSegmentBounds", reinterpret_cast<const void*>(&b2ComputeSegmentBounds)},
        {"b2CreateSegmentShape", reinterpret_cast<const void*>(&b2CreateSegmentShape)},
        {"b2Shape_GetSegment", reinterpret_cast<const void*>(&b2Shape_GetSegment)},
        {"b2UpdateBodyMassData", reinterpret_cast<const void*>(&b2UpdateBodyMassData)},
    };
    const int n = static_cast<int>(sizeof(syms) / sizeof(syms[0]));
    SEXP out = PROTECT(Rf_allocVector(STRSXP, n));
    SEXP nms = PROTECT(Rf_allocVector(STRSXP, n));
    for (int i = 0; i < n; i++) {
        const char* file = "";
#ifndef _WIN32
        Dl_info info;
        if (dladdr(syms[i].addr, &info) && info.dli_fname) file = info.dli_fname;
#endif
        SET_STRING_ELT(out, i, Rf_mkChar(file));
        SET_STRING_ELT(nms, i, Rf_mkChar(syms[i].name));
    }
    Rf_setAttrib(out, R_NamesSymbol, nms);
    UNPROTECT(2);
    return out;
}

// A body dropped from 10 m onto ground with its top at y = 0, stepped as the
// lunar lander is (50 Hz, LL_SUBSTEPS) for 2.4 s: its height at the end.
// Ground: a polygon or a segment (as the lander's terrain). Body: 0 a 1 m box
// (0.5 at rest), filter default or the lander's (category CAT_LANDER, mask
// CAT_GROUND on a CAT_GROUND ground); 1 the lander's hull (its bottom is
// 10 / LL_SCALE below the centre: 0.333 at rest); 2 the hull with its legs.
static double drop_box(bool segment, bool lander_filter, int body = 0) {
    using namespace rltr;
    b2WorldDef wd = b2DefaultWorldDef();
    wd.gravity = {0.0f, -10.0f};
    wd.workerCount = 1;
    b2WorldId world = b2CreateWorld(&wd);

    b2BodyDef gd = b2DefaultBodyDef();
    b2BodyId ground = b2CreateBody(world, &gd);
    b2ShapeDef gs = b2DefaultShapeDef();
    if (lander_filter) gs.filter.categoryBits = CAT_GROUND;
    if (segment) {
        b2Segment seg = {{-50.0f, 0.0f}, {50.0f, 0.0f}};
        b2CreateSegmentShape(ground, &gs, &seg);
    } else {
        b2Polygon gbox = b2MakeOffsetBox(50.0f, 1.0f, {0.0f, -1.0f}, b2Rot_identity);
        b2CreatePolygonShape(ground, &gs, &gbox);
    }

    b2BodyId box;
    if (body == 0) {
        b2BodyDef bd = b2DefaultBodyDef();
        bd.type = b2_dynamicBody;
        bd.position = {0.0f, 10.0f};
        box = b2CreateBody(world, &bd);
        b2ShapeDef bs = b2DefaultShapeDef();
        if (lander_filter) {
            bs.filter.categoryBits = CAT_LANDER;
            bs.filter.maskBits = CAT_GROUND;
        }
        b2Polygon bbox = b2MakeBox(0.5f, 0.5f);
        b2CreatePolygonShape(box, &bs, &bbox);
    } else {
        box = make_hull(world, 0.0, 10.0);
        if (body == 2) {
            b2BodyId legs[2];
            make_legs(world, box, 0.0, 10.0, legs);
        }
    }

    for (int i = 0; i < 120; i++) b2World_Step(world, static_cast<float>(1.0 / LL_FPS), LL_SUBSTEPS);
    const double y = b2Body_GetPosition(box).y;
    b2DestroyWorld(world);
    return y;
}

// Where a polygon-segment collision is lost, in three stages. out[0], out[1]:
// narrow phase, b2CollideSegmentAndPolygon() of a 1 m box at (0, 0.4) and the
// segment (-50, 0)-(50, 0): 2 points, separation -0.1. out[2..5]: the segment
// shape's bounds in a world (lower x, y, upper x, y): -50, 0, 50, 0. out[6]:
// broad phase, the most contacts the world held while a box fell onto the
// segment (as in drop_box()): 1 when a contact was created. Then the steps of
// the segment's bounds: out[7..10] b2ComputeSegmentBounds() with the identity
// transform, no world (-50, 0, 50, 0); out[11..14] the ground body's transform
// p.x, p.y, q.c, q.s (0, 0, 1, 0); out[15..18] the segment as stored in the
// shape (-50, 0, 50, 0); out[19] shapes the broad-phase tree finds in a box
// around the origin (1); out[20..23] the fat AABB the broad phase pairs by and
// out[24..27] the AABB of the shape's tree node (both about -50, 0, 50, 0 plus
// a small margin); out[28..31] b2ComputeSegmentBounds() with the body's
// transform (-50, 0, 50, 0); out[32..35] and out[36..39] the shape's AABB
// recomputed as shape creation does (b2Shape_ComputeFatAABB) with margin 0
// (-50, 0, 50, 0) and 0.02 (-50.02, -0.02, 50.02, 0.02).
static bool count_shape(b2ShapeId, void* context) {
    ++*static_cast<int*>(context);
    return true;
}

static void segment_checks(double out[40]) {
    using namespace rltr;
    b2Segment seg = {{-50.0f, 0.0f}, {50.0f, 0.0f}};
    b2Polygon box = b2MakeBox(0.5f, 0.5f);
    b2Transform xf = {{0.0f, 0.4f}, {1.0f, 0.0f}};
    b2LocalManifold m = b2CollideSegmentAndPolygon(&seg, &box, xf);
    out[0] = m.pointCount;
    out[1] = m.pointCount > 0 ? std::min(m.points[0].separation,
                                         m.pointCount > 1 ? m.points[1].separation : m.points[0].separation)
                              : NA_REAL;

    b2WorldDef wd = b2DefaultWorldDef();
    wd.gravity = {0.0f, -10.0f};
    wd.workerCount = 1;
    b2WorldId world = b2CreateWorld(&wd);
    b2BodyDef gd = b2DefaultBodyDef();
    b2BodyId ground = b2CreateBody(world, &gd);
    b2ShapeDef gs = b2DefaultShapeDef();
    b2ShapeId seg_shape = b2CreateSegmentShape(ground, &gs, &seg);
    const b2AABB bb = b2Shape_GetBounds(seg_shape);
    out[2] = bb.lowerBound.x;
    out[3] = bb.lowerBound.y;
    out[4] = bb.upperBound.x;
    out[5] = bb.upperBound.y;

    const b2WorldTransform id_xf = {{0.0f, 0.0f}, {1.0f, 0.0f}};
    const b2AABB pure = b2ComputeSegmentBounds(&seg, id_xf);
    out[7] = pure.lowerBound.x;
    out[8] = pure.lowerBound.y;
    out[9] = pure.upperBound.x;
    out[10] = pure.upperBound.y;
    const b2WorldTransform gxf = b2Body_GetTransform(ground);
    out[11] = gxf.p.x;
    out[12] = gxf.p.y;
    out[13] = gxf.q.c;
    out[14] = gxf.q.s;
    const b2Segment stored = b2Shape_GetSegment(seg_shape);
    out[15] = stored.point1.x;
    out[16] = stored.point1.y;
    out[17] = stored.point2.x;
    out[18] = stored.point2.y;
    int found = 0;
    const b2AABB probe = {{-1.0f, -1.0f}, {1.0f, 1.0f}};
    b2World_OverlapAABB(world, {0.0f, 0.0f}, probe, b2DefaultQueryFilter(), count_shape, &found);
    out[19] = found;
    const b2AABB fat = b2Shape_GetFatAABB(seg_shape);
    out[20] = fat.lowerBound.x;
    out[21] = fat.lowerBound.y;
    out[22] = fat.upperBound.x;
    out[23] = fat.upperBound.y;
    const b2AABB tree = b2Shape_GetTreeAABB(seg_shape);
    out[24] = tree.lowerBound.x;
    out[25] = tree.lowerBound.y;
    out[26] = tree.upperBound.x;
    out[27] = tree.upperBound.y;
    const b2AABB with_body = b2ComputeSegmentBounds(&seg, gxf);
    const b2AABB re0 = b2Shape_ComputeFatAABB(seg_shape, 0.0f);
    const b2AABB re2 = b2Shape_ComputeFatAABB(seg_shape, 0.02f);   // B2_SPECULATIVE_DISTANCE
    const b2AABB* boxes[3] = {&with_body, &re0, &re2};
    for (int k = 0; k < 3; k++) {
        out[28 + 4 * k] = boxes[k]->lowerBound.x;
        out[29 + 4 * k] = boxes[k]->lowerBound.y;
        out[30 + 4 * k] = boxes[k]->upperBound.x;
        out[31 + 4 * k] = boxes[k]->upperBound.y;
    }

    b2BodyDef bd = b2DefaultBodyDef();
    bd.type = b2_dynamicBody;
    bd.position = {0.0f, 10.0f};
    b2BodyId body = b2CreateBody(world, &bd);
    b2ShapeDef bs = b2DefaultShapeDef();
    b2CreatePolygonShape(body, &bs, &box);
    int max_contacts = 0;
    for (int i = 0; i < 120; i++) {
        b2World_Step(world, static_cast<float>(1.0 / LL_FPS), LL_SUBSTEPS);
        max_contacts = std::max(max_contacts, b2World_GetCounters(world).contactCount);
    }
    out[6] = max_contacts;
    b2DestroyWorld(world);
}

// Build check: Box2D's version and the height of a 1 m box dropped from 10 m
// onto the ground (top at y = 0) after 2 s -- it comes to rest at y = 0.5.
// Then drop_box() heights: a box (0.5 at rest) on segment ground; on polygon
// ground with the lander's filter; on segment ground with the lander's filter;
// the lander's hull on segment ground (0.333); the hull with its legs. Then
// segment_checks(): narrow phase, segment bounds, broad phase.
extern "C" SEXP rlt_box2d_check(void) {
    b2WorldDef wd = b2DefaultWorldDef();
    wd.gravity = {0.0f, -10.0f};
    wd.workerCount = 1;
    b2WorldId world = b2CreateWorld(&wd);

    b2BodyDef gd = b2DefaultBodyDef();
    gd.position = {0.0f, -1.0f};
    b2BodyId ground = b2CreateBody(world, &gd);
    b2ShapeDef sd = b2DefaultShapeDef();
    b2Polygon gbox = b2MakeBox(50.0f, 1.0f);
    b2CreatePolygonShape(ground, &sd, &gbox);

    b2BodyDef bd = b2DefaultBodyDef();
    bd.type = b2_dynamicBody;
    bd.position = {0.0f, 10.0f};
    b2BodyId box = b2CreateBody(world, &bd);
    b2Polygon bbox = b2MakeBox(0.5f, 0.5f);
    b2CreatePolygonShape(box, &sd, &bbox);

    for (int i = 0; i < 120; i++) b2World_Step(world, 1.0f / 60.0f, 4);
    const double y = b2Body_GetPosition(box).y;
    b2DestroyWorld(world);

    b2Version v = b2GetVersion();
    const double y_seg = drop_box(true, false);
    const double y_filt = drop_box(false, true);
    const double y_seg_filt = drop_box(true, true);
    const double y_hull = drop_box(true, true, 1);
    const double y_legs = drop_box(true, true, 2);
    double sc[40];
    segment_checks(sc);
    SEXP out = PROTECT(Rf_allocVector(REALSXP, 49));
    for (int i = 0; i < 40; i++) REAL(out)[9 + i] = sc[i];
    REAL(out)[0] = v.major;
    REAL(out)[1] = v.minor;
    REAL(out)[2] = v.revision;
    REAL(out)[3] = y;
    REAL(out)[4] = y_seg;
    REAL(out)[5] = y_filt;
    REAL(out)[6] = y_seg_filt;
    REAL(out)[7] = y_hull;
    REAL(out)[8] = y_legs;
    UNPROTECT(1);
    return out;
}
