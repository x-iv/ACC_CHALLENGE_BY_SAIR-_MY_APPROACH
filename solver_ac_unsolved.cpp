// =============================================================================
// solver_ac_unsolved.cpp
//
// Standard AC solver for unsolved problems.
//
// Reads:  ac_unsolved.jsonl
// Writes: submission_ac.txt
//
// Target: <x, y | x, y>   (i.e. relators x and y, both trivial)
//
// Compile:
//   clang++ -std=c++17 -O3 -march=native solver_ac_unsolved.cpp -o solver_ac_unsolved.exe
//
// Run:
//   ./solver_ac_unsolved.exe ac_unsolved.jsonl submission_ac.txt
//
// Optional:
//   ./solver_ac_unsolved.exe ac_unsolved.jsonl submission_ac.txt 900 28800
//   (per-problem seconds, total seconds)
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
#include <vector>
#include <chrono>
#include <ctime>

// =============================================================================
// TUNABLE PARAMETERS
// =============================================================================
static const int   DEFAULT_PER_PROBLEM_SECONDS = 600;    // 10 min
static const int   DEFAULT_TOTAL_RUN_SECONDS   = 28800;  // 8 hours
static const long long MAX_STATES              = 20'000'000;
static const int   TRIAL_SECONDS               = 60;     // per weight trial

// =============================================================================
// BASIC TYPES
// =============================================================================
using Word    = std::vector<int8_t>;       // signed letters: 1=x, -1=X, 2=y, -2=Y
using StateAC = std::pair<Word, Word>;     // exactly two relators

// =============================================================================
// WORD OPERATIONS
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
// STANDARD AC MOVES  (14 total)
//
// 0: r0 -> r0^{-1}
// 1: r1 -> r1^{-1}
// 2: r0 -> r0 * r1
// 3: r0 -> r0 * r1^{-1}
// 4: r1 -> r1 * r0
// 5: r1 -> r1 * r0^{-1}
// 6..9:  r0 -> g * r0 * g^{-1}  for g in {x, X, y, Y}
// 10..13: r1 -> g * r1 * g^{-1}  for g in {x, X, y, Y}
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
// STATE HASHING
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
// HEURISTIC
// =============================================================================
int heuristic_ac(const StateAC& s) {
    return (int)(s.first.size() + s.second.size());
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
        return a.f > b.f;   // min-heap on f
    }
};

// =============================================================================
// WEIGHTED A*
// =============================================================================
bool astar_ac(const StateAC& initial,
              const StateAC& target,
              int budget_seconds,
              double weight,
              std::vector<int>& out_moves,
              long long& out_expansions) {

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

    while (!open.empty()) {
        if (elapsed() > budget_seconds) return false;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                now - last_heartbeat).count() >= 5) {
            std::cout << "      [hb] open=" << open.size()
                      << " seen=" << best_g.size()
                      << " elapsed=" << elapsed() << "s\n" << std::flush;
            last_heartbeat = now;
        }

        if ((long long)best_g.size() > MAX_STATES) return false;

        Node top = std::move(const_cast<Node&>(open.top()));
        open.pop();
        out_expansions++;

        if (top.state == target) {
            out_moves = std::move(top.path);
            return true;
        }

        auto it = best_g.find(top.state);
        if (it != best_g.end() && top.g > it->second) continue;

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
    if (rels.size() == 1) rels.push_back(Word{});  // pad to 2
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
// MAIN
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: solver_ac_unsolved.exe <ac_unsolved.jsonl> <submission_ac.txt> "
                     "[per_problem_seconds] [total_seconds]\n";
        return 1;
    }

    std::ios::sync_with_stdio(false);
    std::cout << std::unitbuf;

    std::string input_path  = argv[1];
    std::string output_path = argv[2];
    int per_problem_seconds = (argc >= 4) ? std::stoi(argv[3])
                                          : DEFAULT_PER_PROBLEM_SECONDS;
    int total_run_seconds   = (argc >= 5) ? std::stoi(argv[4])
                                          : DEFAULT_TOTAL_RUN_SECONDS;

    std::ifstream in(input_path);
    if (!in) {
        std::cerr << "cannot open " << input_path << "\n";
        return 1;
    }

    // ---- Load problems ----
    std::vector<Problem> problems;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try { problems.push_back(parse_line(line)); }
        catch (const std::exception& e) {
            std::cerr << "  [warn] parse error: " << e.what() << "\n";
        }
    }

    // Sort shortest first (easier problems solved first)
    std::sort(problems.begin(), problems.end(),
              [](const Problem& a, const Problem& b){
                  return a.total_len < b.total_len;
              });

    std::cout << "Standard AC solver for unsolved problems\n";
    std::cout << "Loaded:             " << problems.size() << " problems\n";
    std::cout << "Per-problem budget: " << per_problem_seconds << "s\n";
    std::cout << "Total run budget:   " << total_run_seconds << "s\n";
    std::cout << "Start:              " << timestamp() << "\n\n";

    std::ofstream out(output_path, std::ios::app);

    auto overall_start = std::chrono::steady_clock::now();
    auto overall_elapsed = [&]() {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - overall_start).count();
    };

    const StateAC target{{1}, {2}};   // <x, y | x, y>

    int solved = 0, attempted = 0;

    for (size_t i = 0; i < problems.size(); ++i) {
        if (overall_elapsed() > total_run_seconds) {
            std::cout << "\n*** Overall time budget reached. ***\n";
            break;
        }

        const Problem& p = problems[i];
        attempted++;

        std::cout << "[" << (i + 1) << "/" << problems.size() << "] "
                  << p.cid << " (len=" << p.total_len << ")\n";

        std::vector<int> best_moves;
        int best_len = 999999;
        bool found = false;

        // ---- Trial with multiple weights ----
        std::vector<double> weights = {1.0, 2.0, 4.0};
        for (double w : weights) {
            std::vector<int> moves;
            long long exp = 0;
            bool ok = astar_ac(p.initial, target, TRIAL_SECONDS, w, moves, exp);
            if (ok) {
                std::cout << "  trial w=" << w << " SOLVED ("
                          << moves.size() << " moves, "
                          << exp << " exp)\n";
                if ((int)moves.size() < best_len) {
                    best_len = (int)moves.size();
                    best_moves = moves;
                    found = true;
                }
            } else {
                std::cout << "  trial w=" << w << " failed ("
                          << exp << " exp)\n";
            }
        }

        // ---- Commit remaining time to best weight ----
        if (found) {
            int remaining = per_problem_seconds - TRIAL_SECONDS * (int)weights.size();
            if (remaining > 0) {
                std::vector<int> moves;
                long long exp = 0;
                bool ok = astar_ac(p.initial, target, remaining, 2.0, moves, exp);
                if (ok && (int)moves.size() < best_len) {
                    best_len = (int)moves.size();
                    best_moves = moves;
                }
            }
        } else {
            // No solution found in trials — give full remaining budget to w=2.0
            std::vector<int> moves;
            long long exp = 0;
            bool ok = astar_ac(p.initial, target,
                               per_problem_seconds - TRIAL_SECONDS * (int)weights.size(),
                               2.0, moves, exp);
            if (ok) {
                best_len = (int)moves.size();
                best_moves = moves;
                found = true;
            }
        }

        // ---- Report ----
        if (found) {
            std::cout << "  => AC SOLVED (" << best_moves.size() << " moves)\n";
            out << p.cid << ": [";
            for (size_t k = 0; k < best_moves.size(); ++k) {
                if (k) out << ", ";
                out << best_moves[k];
            }
            out << "]\n";
            out.flush();
            solved++;
        } else {
            std::cout << "  => AC FAILED\n";
        }
    }

    std::cout << "\n==================================================\n";
    std::cout << "Done.\n";
    std::cout << "Solved:    " << solved << "/" << attempted << "\n";
    std::cout << "End:       " << timestamp() << "\n";
    std::cout << "Output:    " << output_path << "\n";
    std::cout << "==================================================\n";

    return 0;
}