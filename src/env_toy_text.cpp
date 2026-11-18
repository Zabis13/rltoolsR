// Gymnasium toy_text environments as transition tables (env_tabular.h),
// ported from https://github.com/Farama-Foundation/Gymnasium
// (gymnasium/envs/toy_text). Each table is built as in the Python source.

#include <algorithm>

#include "env_tabular.h"

namespace rltr {

namespace {

// ------------------------------------------------------------- frozen lake

// Gymnasium FrozenLake-v1 (toy_text/frozen_lake.py) with the standard maps.
// Actions 0 left, 1 down, 2 right, 3 up. Slippery: the intended direction
// and the two perpendicular ones, 1/3 each. H and G end the episode, G gives 1.
const char* const FROZEN_LAKE_4X4[] = {"SFFF", "FHFH", "FFFH", "HFFG"};
const char* const FROZEN_LAKE_8X8[] = {"SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
                                       "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"};

TabularTable frozen_lake_table(const char* const* desc, int nrow, bool slippery) {
    const int ncol = nrow;
    TabularTable t;
    t.S = nrow * ncol;
    t.A = 4;
    t.K = slippery ? 3 : 1;
    t.P.resize(static_cast<size_t>(t.S) * t.A * t.K);
    t.isd.assign(t.S, 0);

    auto letter = [&](int row, int col) { return desc[row][col]; };
    auto move = [&](int row, int col, int a, int& r, int& c) {
        r = row; c = col;
        if (a == 0) c = std::max(col - 1, 0);
        else if (a == 1) r = std::min(row + 1, nrow - 1);
        else if (a == 2) c = std::min(col + 1, ncol - 1);
        else r = std::max(row - 1, 0);
    };

    int n_start = 0;
    for (int row = 0; row < nrow; row++)
        for (int col = 0; col < ncol; col++) if (letter(row, col) == 'S') n_start++;

    for (int row = 0; row < nrow; row++) {
        for (int col = 0; col < ncol; col++) {
            const int s = row * ncol + col;
            const char l = letter(row, col);
            if (l == 'S') t.isd[s] = 1.0 / n_start;
            for (int a = 0; a < 4; a++) {
                if (l == 'G' || l == 'H') {
                    t.at(s, a, 0) = {1.0, s, 0.0, true};
                    continue;
                }
                const int dirs[3] = {(a + 3) % 4, a, (a + 1) % 4};
                for (int k = 0; k < t.K; k++) {
                    const int b = slippery ? dirs[k] : a;
                    int r, c;
                    move(row, col, b, r, c);
                    const char nl = letter(r, c);
                    t.at(s, a, k) = {slippery ? 1.0 / 3.0 : 1.0, r * ncol + c,
                                     nl == 'G' ? 1.0 : 0.0, nl == 'G' || nl == 'H'};
                }
            }
        }
    }
    return t;
}

Env* make_frozen_lake_impl(int64_t seed, const std::string& name, bool big, bool slippery, bool onehot) {
    const int nrow = big ? 8 : 4;
    return new TabularEnv(seed, name, big ? 200 : 100,
                          frozen_lake_table(big ? FROZEN_LAKE_8X8 : FROZEN_LAKE_4X4, nrow, slippery),
                          onehot);
}

// ----------------------------------------------------------- cliff walking

// Gymnasium CliffWalking (toy_text/cliffwalking.py, is_slippery = False):
// 4 x 12 grid, start (3, 0), goal (3, 11), the cliff (3, 1..10). Actions 0 up,
// 1 right, 2 down, 3 left; moves are clipped to the grid. Reward -1 per step;
// stepping into the cliff gives -100 and returns to the start without ending
// the episode; reaching the goal ends it. Gymnasium has no step limit here,
// we use 200.
constexpr int CLIFF_ROWS = 4, CLIFF_COLS = 12, CLIFF_MAX_STEPS = 200;

TabularTable cliff_walking_table() {
    TabularTable t;
    t.S = CLIFF_ROWS * CLIFF_COLS;
    t.A = 4;
    t.K = 1;
    t.P.resize(static_cast<size_t>(t.S) * t.A);
    t.isd.assign(t.S, 0);
    const int start = (CLIFF_ROWS - 1) * CLIFF_COLS;
    t.isd[start] = 1;
    const int dr[4] = {-1, 0, 1, 0}, dc[4] = {0, 1, 0, -1};
    for (int row = 0; row < CLIFF_ROWS; row++) {
        for (int col = 0; col < CLIFF_COLS; col++) {
            const int s = row * CLIFF_COLS + col;
            for (int a = 0; a < 4; a++) {
                const int r = std::min(std::max(row + dr[a], 0), CLIFF_ROWS - 1);
                const int c = std::min(std::max(col + dc[a], 0), CLIFF_COLS - 1);
                const bool cliff = r == CLIFF_ROWS - 1 && c >= 1 && c <= CLIFF_COLS - 2;
                if (cliff) {
                    t.at(s, a, 0) = {1.0, start, -100.0, false};
                } else {
                    const bool goal = r == CLIFF_ROWS - 1 && c == CLIFF_COLS - 1;
                    t.at(s, a, 0) = {1.0, r * CLIFF_COLS + c, -1.0, goal};
                }
            }
        }
    }
    return t;
}

// -------------------------------------------------------------------- taxi

// Gymnasium Taxi-v3 (toy_text/taxi.py, is_rainy = fickle_passenger = False):
// 5 x 5 grid with walls, passenger at one of R, G, Y, B or in the taxi (4),
// destination one of R, G, Y, B. State ((row * 5 + col) * 5 + pass) * 4 + dest.
// Actions 0 south, 1 north, 2 east, 3 west, 4 pickup, 5 dropoff. Reward -1
// per step, 20 for the delivery (ends the episode), -10 for an illegal pickup
// or dropoff. Initial state uniform over the passenger not in the taxi and
// not at the destination (300 states). 200 steps.
const char* const TAXI_MAP[] = {
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+"};
constexpr int TAXI_MAX_STEPS = 200;

TabularTable taxi_table() {
    const int locs[4][2] = {{0, 0}, {0, 4}, {4, 0}, {4, 3}};
    auto encode = [](int row, int col, int pass, int dest) { return ((row * 5 + col) * 5 + pass) * 4 + dest; };
    TabularTable t;
    t.S = 500;
    t.A = 6;
    t.K = 1;
    t.P.resize(static_cast<size_t>(t.S) * t.A);
    t.isd.assign(t.S, 0);
    int n_start = 0;
    for (int row = 0; row < 5; row++) {
        for (int col = 0; col < 5; col++) {
            int at = -1;  // index of the location the taxi is at
            for (int i = 0; i < 4; i++) if (locs[i][0] == row && locs[i][1] == col) at = i;
            for (int pass = 0; pass < 5; pass++) {
                for (int dest = 0; dest < 4; dest++) {
                    const int s = encode(row, col, pass, dest);
                    if (pass < 4 && pass != dest) { t.isd[s] = 1; n_start++; }
                    for (int a = 0; a < 6; a++) {
                        int nr = row, nc = col, np = pass;
                        double reward = -1;
                        bool terminated = false;
                        if (a == 0) nr = std::min(row + 1, 4);
                        else if (a == 1) nr = std::max(row - 1, 0);
                        else if (a == 2 && TAXI_MAP[1 + row][2 * col + 2] == ':') nc = std::min(col + 1, 4);
                        else if (a == 3 && TAXI_MAP[1 + row][2 * col] == ':') nc = std::max(col - 1, 0);
                        else if (a == 4) {
                            if (pass < 4 && at == pass) np = 4; else reward = -10;
                        } else if (a == 5) {
                            if (at == dest && pass == 4) { np = dest; terminated = true; reward = 20; }
                            else if (at >= 0 && pass == 4) np = at;
                            else reward = -10;
                        }
                        t.at(s, a, 0) = {1.0, encode(nr, nc, np, dest), reward, terminated};
                    }
                }
            }
        }
    }
    for (double& p : t.isd) p /= n_start;
    return t;
}

// --------------------------------------------------------------- blackjack

// Gymnasium Blackjack-v1 (toy_text/blackjack.py, natural = sab = False) as an
// exact table: the deck is infinite (1-9 with 1/13 each, 10 with 4/13), so
// (player sum, dealer's card, usable ace) is a sufficient state -- an ace that
// is not usable never becomes usable again. Actions 0 stick, 1 hit; no
// automatic stick at 21. Hit: 10 outcomes by the card, a bust ends the episode
// with -1 (the observation then shows the bust sum, ace 0). Stick: win, draw
// or loss (+1, 0, -1) against the dealer, who draws while his sum (ace as 11
// when usable) is below 17, so he stands on soft 17; the observation stays.
// The state is the observation, s = (sum * 11 + dealer) * 2 + ace over the
// whole cube 32 x 11 x 2; unreachable and bust states loop on themselves.
// Gymnasium has no step limit; an episode ends within 12 steps, 100 is a
// limit that never triggers.
constexpr int BJ_MAX_STEPS = 100;

double bj_card_prob(int c) { return c == 10 ? 4.0 / 13.0 : 1.0 / 13.0; }

// sum with a usable ace counted as 11
int bj_sum(int raw, bool ace) { return ace && raw + 10 <= 21 ? raw + 10 : raw; }

// Distribution of the dealer's final score (index 0: bust, 17..21) from a hand
// with raw sum `raw` and `ace`.
void bj_dealer(int raw, bool ace, double p, double* dist) {
    const int s = bj_sum(raw, ace);
    if (s >= 17) {
        dist[s > 21 ? 0 : s] += p;
        return;
    }
    for (int c = 1; c <= 10; c++) bj_dealer(raw + c, ace || c == 1, p * bj_card_prob(c), dist);
}

TabularTable blackjack_table(bool onehot) {
    auto encode = [](int sum, int dealer, int ace) { return (sum * 11 + dealer) * 2 + ace; };
    TabularTable t;
    t.S = 32 * 11 * 2;
    t.A = 2;
    t.K = 10;
    t.P.resize(static_cast<size_t>(t.S) * t.A * t.K);
    t.isd.assign(t.S, 0);

    // dealer's final score by his open card
    double dealer[11][22] = {};
    for (int d = 1; d <= 10; d++)
        for (int h = 1; h <= 10; h++) bj_dealer(d + h, d == 1 || h == 1, bj_card_prob(h), dealer[d]);

    for (int sum = 0; sum < 32; sum++) {
        for (int d = 0; d <= 10; d++) {
            for (int ace = 0; ace <= 1; ace++) {
                const int s = encode(sum, d, ace);
                const bool valid = d >= 1 && sum >= 4 && sum <= 21 && (ace == 0 || sum >= 12);
                if (!valid) {
                    for (int a = 0; a < 2; a++) t.at(s, a, 0) = {1.0, s, 0.0, true};
                    continue;
                }
                // stick
                double win = dealer[d][0], draw = 0, loss = 0;
                for (int ds = 17; ds <= 21; ds++) {
                    if (ds < sum) win += dealer[d][ds];
                    else if (ds == sum) draw += dealer[d][ds];
                    else loss += dealer[d][ds];
                }
                t.at(s, 0, 0) = {win, s, 1.0, true};
                t.at(s, 0, 1) = {draw, s, 0.0, true};
                t.at(s, 0, 2) = {loss, s, -1.0, true};
                for (int k = 3; k < t.K; k++) t.at(s, 0, k) = {0.0, s, 0.0, true};
                // hit
                for (int c = 1; c <= 10; c++) {
                    const int raw = ace ? sum - 10 + c : sum + c;
                    const bool usable = (ace || c == 1) && raw + 10 <= 21;
                    const int ns = usable ? raw + 10 : raw;
                    t.at(s, 1, c - 1) = ns > 21 ? Outcome{bj_card_prob(c), encode(ns, d, 0), -1.0, true}
                                                : Outcome{bj_card_prob(c), encode(ns, d, usable), 0.0, false};
                }
            }
        }
    }

    // two cards for the player, the dealer's open card
    for (int c1 = 1; c1 <= 10; c1++)
        for (int c2 = 1; c2 <= 10; c2++) {
            const bool ace = c1 == 1 || c2 == 1;
            const int raw = c1 + c2;
            const bool usable = ace && raw + 10 <= 21;
            for (int d = 1; d <= 10; d++)
                t.isd[encode(bj_sum(raw, ace), d, usable)] +=
                    bj_card_prob(c1) * bj_card_prob(c2) * bj_card_prob(d);
        }

    // observations: (sum, dealer, ace) as Gymnasium's tuple, or its three
    // one-hots concatenated (32 + 11 + 2)
    t.obs_dim = onehot ? 45 : 3;
    t.obs.assign(static_cast<size_t>(t.S) * t.obs_dim, 0.0f);
    Space& o = t.obs_space;
    if (onehot) {
        o.type = SpaceType::Box;
        o.shape = {45};
        o.low.assign(45, 0);
        o.high.assign(45, 1);
    } else {
        o.type = SpaceType::MultiDiscrete;
        o.dtype = DType::Int32;
        o.nvec = {32, 11, 2};
    }
    for (int sum = 0; sum < 32; sum++)
        for (int d = 0; d <= 10; d++)
            for (int ace = 0; ace <= 1; ace++) {
                float* row = t.obs.data() + static_cast<size_t>(encode(sum, d, ace)) * t.obs_dim;
                if (onehot) {
                    row[sum] = 1; row[32 + d] = 1; row[43 + ace] = 1;
                } else {
                    row[0] = static_cast<float>(sum); row[1] = static_cast<float>(d); row[2] = static_cast<float>(ace);
                }
            }
    return t;
}

}  // namespace

// names: frozen_lake[_8x8][_det][_onehot], cliff_walking[_onehot], taxi[_onehot],
// blackjack[_onehot]
Env* make_toy_text(const std::string& name, int64_t seed) {
    if (name == "blackjack" || name == "blackjack_onehot") {
        const bool onehot = name == "blackjack_onehot";
        return new TabularEnv(seed, name, BJ_MAX_STEPS, blackjack_table(onehot), onehot);
    }
    if (name == "taxi" || name == "taxi_onehot") {
        return new TabularEnv(seed, name, TAXI_MAX_STEPS, taxi_table(), name == "taxi_onehot");
    }
    if (name == "cliff_walking" || name == "cliff_walking_onehot") {
        return new TabularEnv(seed, name, CLIFF_MAX_STEPS, cliff_walking_table(),
                              name == "cliff_walking_onehot");
    }
    const std::string base = "frozen_lake";
    if (name.compare(0, base.size(), base) != 0) return nullptr;
    std::string rest = name.substr(base.size());
    auto take = [&](const std::string& suffix) {
        if (rest.compare(0, suffix.size(), suffix) == 0) { rest = rest.substr(suffix.size()); return true; }
        return false;
    };
    const bool big = take("_8x8");
    const bool det = take("_det");
    const bool onehot = take("_onehot");
    if (!rest.empty()) return nullptr;
    return make_frozen_lake_impl(seed, name, big, !det, onehot);
}

}  // namespace rltr
