// =============================================================================
// solver_ac_greedy.cpp
//
// Greedy AC solver with iterative time budget + state cap + resume.
//
// For each problem:
//   - Run A* with w=5.0 (very greedy) for 30s
//   - If fail, try 60s, then 90s, ... up to max_per_problem
//   - Abort iteration early if state cap or stall detected
//
// Target: <x, y | x, y>
//
// Compile (Windows clang):
//   clang++ -std=c++17 -O3 -march=native solver_ac_greedy.cpp -o solver_ac_greedy.exe
//
// Run:
//   solver_ac_greedy.exe ac_unsolved.jsonl submission_ac.txt
//
// Optional:
//   solver_ac_greedy.exe ac_unsolved.jsonl submission_ac.txt 30 300 28800
//   (step_seconds, max_per_problem, total_seconds)
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <ctime>

// =============================================================================
// TUNABLE PARAMETERS
// =============================================================================
static const int DEFAULT_STEP_SECONDS          = 30;
static const int DEFAULT_MAX_PER_PROBLEM       = 300;
static const int DEFAULT_TOTAL_RUN_SECONDS     = 28800;

// Greedy weight: higher = greedier = faster but maybe not shortest.
static const double GREEDY_WEIGHT              = 5.0;

// Hard cap on states per A* iteration. Prevents memory blowup / freeze.
static const long long MAX_STATES_PER_ITER     = 4'000'000;

// If we expand this many nodes without reaching a new best, abort iteration.
static const long long STALL_LIMIT             = 500'000;

// How often to check time / print heartbeat (in expansions)
static const long long TIME_CHECK_EVERY        = 50'000;

// =============================================================================
// TYPES
// =============================================================================
using Word    = std::vector<int8_t>;
using StateAC = std::pair<Word, Word>;

// =============================================================================
// WORD OPS
// =============================================================================
Word free_reduce(const Word& w) {
    Word out; out.reserve(w.size());
    for (int8_t a : w) {
        if (!out.empty() && out.back() == -a) out.pop_back();
        else                                  out.push_back(a);
    }
    return out;
}

Word invert_word(const Word& w) {
    Word out(w.rbegin(), w.rend());
    for (auto& a : out) a = -a;
    return out;
}

static const int8_t GENS[4] = {1, -1, 2, -2};

// =============================================================================
// STANDARD AC MOVES (14)
// =============================================================================
StateAC apply_ac_move(const StateAC& s, int m) {
    const Word& r0 = s.first;
    const Word& r1 = s.second;
    switch (m) {
        case 0:  return {invert_word(r0), r1};
        case 1:  return {r0, invert_word(r1)};
        case 2:  { Word x = r0; x.insert(x.end(), r1.begin(), r1.end());
                   return {free_reduce(x), r1}; }
        case 3:  { Word inv = invert_word(r1);
                   Word x = r0; x.insert(x.end(), inv.begin(), inv.end());
                   return {free_reduce(x), r1}; }
        case 4:  { Word x = r1; x.insert(x.end(), r0.begin(), r0.end());
                   return {r0, free_reduce(x)}; }
        case 5:  { Word inv = invert_word(r0);
                   Word x = r1; x.insert(x.end(), inv.begin(), inv.end());
                   return {r0, free_reduce(x)}; }
        case 6: case 7: case 8: case 9: {
            int8_t g = GENS[m - 6];
            Word x; x.push_back(g);
            x.insert(x.end(), r0.begin(), r0.end());
            x.push_back(-g);
            return {free_reduce(x), r1};
        }
        case 10: case 11: case 12: case 13: {
            int8_t g = GENS[m - 10];
            Word x; x.push_back(g);
            x.insert(x.end(), r1.begin(), r1.end());
            x.push_back(-g);
            return {r0, free_reduce(x)};
        }
    }
    throw std::runtime_error("bad AC move id");
}
const int NUM_AC_MOVES = 14;

// =============================================================================
// HASH
// =============================================================================
struct StateHashAC {
    size_t operator()(const StateAC& s) const {
        size_t h = 1469598103934665603ULL;
        for (auto a : s.first)  { h ^= (size_t)(a + 10); h *= 1099511628211ULL; }
        h ^= 0xFFFFULL; h *= 1099511628211ULL;
        for (auto a : s.second) { h ^= (size_t)(a + 10); h *= 1099511628211ULL; }
        return h;
    }
};

// =============================================================================
// GREEDY HEURISTIC
//
// Total relator length, PLUS a bonus for how close each relator is to
// being a single generator (which is the goal: r0 = x, r1 = y).
// =============================================================================
int heuristic_ac(const StateAC& s) {
    int len = (int)(s.first.size() + s.second.size());

    // Bonus: prefer relators that are already a single letter
    int bonus = 0;
    if (s.first.size() == 1)  bonus += 2;
    if (s.second.size() == 1) bonus += 2;

    // Penalize empty relators (they still count as 0 length, but we
    // want x and y specifically, not empty)
    if (s.first.empty())  bonus += 1;
    if (s.second.empty()) bonus += 1;

    return len + bonus;
}

// =============================================================================
// A* NODE
// =============================================================================
struct Node {
    int f, g;
    StateAC state;
    std::vector<int> path;
};

struct CompareNode {
    bool operator()(const Node& a, const Node& b) const {
        return a.f > b.f;
    }
};

// =============================================================================
// GREEDY A* WITH STATE CAP + STALL DETECTION
// =============================================================================
bool astar_ac(const StateAC& initial,
              const StateAC& target,
              int budget_seconds,
              double weight,
              std::vector<int>& out_moves,
              long long& out_expansions,
              long long& out_states_seen) {

    auto t0 = std::chrono::steady_clock::now();
    auto last_heartbeat = t0;
    auto elapsed = [&]() {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    std::priority_queue<Node, std::vector<Node>, CompareNode> open;
    std::unordered_map<StateAC, int, StateHashAC> best_g;

    Node start;
    start.g = 0;
    start.f = (int)(weight * heuristic_ac(initial));
    start.state = initial;
    open.push(std::move(start));
    best_g[initial] = 0;

    out_expansions = 0;
    out_states_seen = 0;

    int best_h = heuristic_ac(initial);
    long long last_improvement_exp = 0;

    while (!open.empty()) {
        // ---- Time check (every loop, cheap) ----
        if (elapsed() > budget_seconds) {
            out_states_seen = (long long)best_g.size();
            return false;
        }

        // ---- Heartbeat every 10s ----
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                now - last_heartbeat).count() >= 10) {
            std::cout << "        [hb] open=" << open.size()
                      << " seen=" << best_g.size()
                      << " exp=" << out_expansions
                      << " elapsed=" << elapsed() << "s\n" << std::flush;
            last_heartbeat = now;
        }

        // ---- State cap: abort if too big ----
        if ((long long)best_g.size() > MAX_STATES_PER_ITER) {
            out_states_seen = (long long)best_g.size();
            return false;
        }

        // ---- Stall detection ----
        if (out_expansions - last_improvement_exp > STALL_LIMIT) {
            // no improvement in a long time; abort this iteration
            out_states_seen = (long long)best_g.size();
            return false;
        }

        Node top = std::move(const_cast<Node&>(open.top()));
        open.pop();
        out_expansions++;

        // ---- Goal check ----
        if (top.state == target) {
            out_moves = std::move(top.path);
            out_states_seen = (long long)best_g.size();
            return true;
        }

        // ---- Stale entry check ----
        auto it = best_g.find(top.state);
        if (it != best_g.end() && top.g > it->second) continue;

        // ---- Track improvement ----
        int h = heuristic_ac(top.state);
        if (h < best_h) {
            best_h = h;
            last_improvement_exp = out_expansions;
        }

        // ---- Expand ----
        for (int m = 0; m < NUM_AC_MOVES; ++m) {
            StateAC ns = apply_ac_move(top.state, m);
            int ng = top.g + 1;
            auto nit = best_g.find(ns);
            if (nit == best_g.end() || ng < nit->second) {
                best_g[ns] = ng;
                Node n;
                n.g = ng;
                n.f = ng + (int)(weight * heuristic_ac(ns));
                n.state = ns;
                n.path = top.path;
                n.path.push_back(m);
                open.push(std::move(n));
            }
        }
    }
    out_states_seen = (long long)best_g.size();
    return false;
}

// =============================================================================
// PARSER
// =============================================================================
int8_t char_to_int(char c) {
    switch (c) {
        case 'x': return 1;
        case 'X': return -1;
        case 'y': return 2;
        case 'Y': return -2;
    }
    throw std::runtime_error(std::string("bad char: ") + c);
}

std::string encode(const std::string& w) {
    std::string out;
    for (size_t i = 0; i < w.size(); ++i) {
        if (i + 3 < w.size() && w[i+1] == '^' && w[i+2] == '-' && w[i+3] == '1') {
            out.push_back(toupper(w[i])); i += 3;
        } else if (w[i] != ' ') out.push_back(w[i]);
    }
    return out;
}

Word parse_word(const std::string& s) {
    std::string e = encode(s);
    Word out; out.reserve(e.size());
    for (char c : e) out.push_back(char_to_int(c));
    return out;
}

struct Problem {
    std::string cid;
    StateAC initial;
    int total_len;
};

Problem parse_line(const std::string& line) {
    Problem p;
    size_t cid_pos  = line.find("\"challenge_id\"");
    size_t desc_pos = line.find("\"description\"");
    if (cid_pos == std::string::npos || desc_pos == std::string::npos)
        throw std::runtime_error("missing field");

    size_t cid_start = line.find('"', cid_pos + 14);
    size_t cid_end   = line.find('"', cid_start + 1);
    p.cid = line.substr(cid_start + 1, cid_end - cid_start - 1);

    size_t desc_start = line.find('"', desc_pos + 13);
    size_t desc_end   = line.rfind('"');
    std::string desc  = line.substr(desc_start + 1, desc_end - desc_start - 1);

    size_t bar = desc.find('|');
    if (bar == std::string::npos) throw std::runtime_error("bad desc");
    std::string rels_part = desc.substr(bar + 1);
    size_t gt = rels_part.find('>');
    if (gt != std::string::npos) rels_part = rels_part.substr(0, gt);

    std::vector<Word> rels;
    std::stringstream ss(rels_part);
    std::string chunk;
    while (std::getline(ss, chunk, ';')) {
        while (!chunk.empty() && chunk.front() == ' ') chunk.erase(0, 1);
        while (!chunk.empty() && (chunk.back() == ' ' || chunk.back() == '.'))
            chunk.pop_back();
        size_t eq = chunk.find("= 1");
        if (eq != std::string::npos) chunk = chunk.substr(0, eq);
        while (!chunk.empty() && chunk.back() == ' ') chunk.pop_back();
        if (!chunk.empty()) rels.push_back(parse_word(chunk));
    }
    if (rels.empty()) throw std::runtime_error("no relators");
    if (rels.size() == 1) rels.push_back(Word{});
    p.initial = {rels[0], rels[1]};
    p.total_len = (int)(rels[0].size() + rels[1].size());
    return p;
}

std::string timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

// =============================================================================
// LOAD ALREADY-SOLVED IDS (for resume)
// =============================================================================
std::unordered_set<std::string> load_solved_ids(const std::string& path) {
    std::unordered_set<std::string> solved;
    std::ifstream in(path);
    if (!in) return solved;
    std::string line;
    while (std::getline(in, line)) {
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string cid = line.substr(0, colon);
            while (!cid.empty() && (cid.back() == ' ' || cid.back() == '\t'))
                cid.pop_back();
            if (!cid.empty()) solved.insert(cid);
        }
    }
    return solved;
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: solver_ac_greedy.exe <ac_unsolved.jsonl> <submission_ac.txt> "
                     "[step_seconds] [max_per_problem] [total_seconds]\n";
        return 1;
    }

    std::ios::sync_with_stdio(false);
    std::cout << std::unitbuf;

    std::string input_path  = argv[1];
    std::string output_path = argv[2];
    int step_seconds        = (argc >= 4) ? std::stoi(argv[3]) : DEFAULT_STEP_SECONDS;
    int max_per_problem     = (argc >= 5) ? std::stoi(argv[4]) : DEFAULT_MAX_PER_PROBLEM;
    int total_run_seconds   = (argc >= 6) ? std::stoi(argv[5]) : DEFAULT_TOTAL_RUN_SECONDS;

    // ---- Load problems ----
    std::ifstream in(input_path);
    if (!in) {
        std::cerr << "cannot open " << input_path << "\n";
        return 1;
    }

    std::vector<Problem> problems;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try { problems.push_back(parse_line(line)); }
        catch (const std::exception& e) {
            std::cerr << "  [warn] parse error: " << e.what() << "\n";
        }
    }

    std::sort(problems.begin(), problems.end(),
              [](const Problem& a, const Problem& b){
                  return a.total_len < b.total_len;
              });

    auto already_solved = load_solved_ids(output_path);

    std::cout << "Greedy AC solver (iterative time budget)\n";
    std::cout << "Loaded:               " << problems.size() << " problems\n";
    std::cout << "Already solved:       " << already_solved.size() << "\n";
    std::cout << "Step seconds:         " << step_seconds << "\n";
    std::cout << "Max per problem:      " << max_per_problem << "s\n";
    std::cout << "Total run budget:     " << total_run_seconds << "s\n";
    std::cout << "Greedy weight:        " << GREEDY_WEIGHT << "\n";
    std::cout << "Max states per iter:  " << MAX_STATES_PER_ITER << "\n";
    std::cout << "Stall limit:          " << STALL_LIMIT << "\n";
    std::cout << "Start:                " << timestamp() << "\n\n";

    std::ofstream out(output_path, std::ios::app);

    auto overall_start = std::chrono::steady_clock::now();
    auto overall_elapsed = [&]() {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - overall_start).count();
    };

    const StateAC target{{1}, {2}};

    int solved = 0, attempted = 0, skipped = 0;

    for (size_t i = 0; i < problems.size(); ++i) {
        if (overall_elapsed() > total_run_seconds) {
            std::cout << "\n*** Overall time budget reached. ***\n";
            break;
        }

        const Problem& p = problems[i];

        if (already_solved.count(p.cid)) {
            skipped++;
            continue;
        }

        attempted++;

        std::cout << "[" << (i + 1) << "/" << problems.size() << "] "
                  << p.cid << " (len=" << p.total_len << ")\n";

        bool found = false;
        std::vector<int> best_moves;
        int best_len = 999999;

        // ---- Iterative time loop ----
        for (int budget = step_seconds;
             budget <= max_per_problem;
             budget += step_seconds) {

            if (overall_elapsed() > total_run_seconds) {
                std::cout << "  *** Overall budget hit during iterative loop.\n";
                break;
            }

            std::cout << "  --- Iteration: " << budget << "s budget (w="
                      << GREEDY_WEIGHT << ") ---\n";

            std::vector<int> moves;
            long long exp = 0, seen = 0;
            bool ok = astar_ac(p.initial, target, budget, GREEDY_WEIGHT,
                               moves, exp, seen);

            if (ok) {
                std::cout << "  => SOLVED at " << budget << "s ("
                          << moves.size() << " moves, "
                          << exp << " exp, "
                          << seen << " seen)\n";
                if ((int)moves.size() < best_len) {
                    best_len = (int)moves.size();
                    best_moves = moves;
                    found = true;
                }
                break;
            } else {
                std::cout << "  => failed at " << budget << "s ("
                          << exp << " exp, "
                          << seen << " seen)\n";
            }
        }

        if (found) {
            out << p.cid << ": [";
            for (size_t k = 0; k < best_moves.size(); ++k) {
                if (k) out << ", ";
                out << best_moves[k];
            }
            out << "]\n";
            out.flush();
            solved++;
            already_solved.insert(p.cid);
        } else {
            std::cout << "  => AC FAILED after " << max_per_problem << "s\n";
        }
    }

    std::cout << "\n==================================================\n";
    std::cout << "Done.\n";
    std::cout << "Solved:    " << solved << "\n";
    std::cout << "Skipped:   " << skipped << " (already in submission)\n";
    std::cout << "Attempted: " << attempted << "\n";
    std::cout << "End:       " << timestamp() << "\n";
    std::cout << "Output:    " << output_path << "\n";
    std::cout << "==================================================\n";

    return 0;
}