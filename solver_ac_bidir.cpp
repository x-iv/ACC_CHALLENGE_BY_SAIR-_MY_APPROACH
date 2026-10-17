// =============================================================================
// Bidirectional A* solver for Andrews-Curtis problems.
//
// Forward search from initial state + backward search from target.
// Meet in the middle when frontiers intersect.
//
// Compile:
//   clang++ -std=c++17 -O3 -march=native -D_CRT_SECURE_NO_WARNINGS solver_ac_bidir.cpp -o solver_ac_bidir.exe
//
// Run:
//   solver_ac_bidir.exe ac_unsolved.jsonl submission_bidir.txt
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

using Word  = std::vector<int8_t>;
using State = std::pair<Word, Word>;

// =============================================================================
// TUNABLE PARAMETERS
// =============================================================================
static const int    STEP_SECONDS         = 30;
static const int    MAX_PER_PROBLEM      = 120;
static const int    TOTAL_RUN_SECONDS    = 14400;   // 4 hours
static const long long MAX_STATES_FWD    = 5'000'000;
static const long long MAX_STATES_BWD    = 5'000'000;
static const double WEIGHT_FWD           = 2.0;
static const double WEIGHT_BWD           = 2.0;

// =============================================================================
// CORE SEMANTICS
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

State apply_move(const State& s, int m) {
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
    throw std::runtime_error("bad move id");
}

struct StateHash {
    size_t operator()(const State& s) const {
        size_t h = 1469598103934665603ULL;
        for (auto a : s.first)  { h ^= (size_t)(a + 10); h *= 1099511628211ULL; }
        h ^= 0xFFFFULL; h *= 1099511628211ULL;
        for (auto a : s.second) { h ^= (size_t)(a + 10); h *= 1099511628211ULL; }
        return h;
    }
};

// =============================================================================
// HEURISTIC (same for both directions)
// =============================================================================
int heuristic(const State& s) {
    return (int)(s.first.size() + s.second.size());
}

// =============================================================================
// A* NODE
// =============================================================================
struct Node {
    int f, g;
    State state;
    std::vector<int> path;
};

struct CompareNode {
    bool operator()(const Node& a, const Node& b) const {
        return a.f > b.f;
    }
};

// =============================================================================
// BIDIRECTIONAL A*
// =============================================================================
bool bidirectional_astar(const State& initial,
                         const State& target,
                         int budget_seconds,
                         std::vector<int>& out_moves,
                         long long& out_expansions) {

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    // Forward search
    std::priority_queue<Node, std::vector<Node>, CompareNode> open_f;
    std::unordered_map<State, int, StateHash> best_g_f;
    std::unordered_map<State, std::vector<int>, StateHash> path_f;

    // Backward search
    std::priority_queue<Node, std::vector<Node>, CompareNode> open_b;
    std::unordered_map<State, int, StateHash> best_g_b;
    std::unordered_map<State, std::vector<int>, StateHash> path_b;

    Node start_f;
    start_f.g = 0;
    start_f.f = (int)(WEIGHT_FWD * heuristic(initial));
    start_f.state = initial;
    open_f.push(start_f);
    best_g_f[initial] = 0;
    path_f[initial] = {};

    Node start_b;
    start_b.g = 0;
    start_b.f = (int)(WEIGHT_BWD * heuristic(target));
    start_b.state = target;
    open_b.push(start_b);
    best_g_b[target] = 0;
    path_b[target] = {};

    out_expansions = 0;

    while (!open_f.empty() || !open_b.empty()) {
        if (elapsed() > budget_seconds) return false;

        // ---- Expand forward ----
        if (!open_f.empty()) {
            Node top = open_f.top(); open_f.pop();
            out_expansions++;

            auto it = best_g_f.find(top.state);
            if (it != best_g_f.end() && top.g > it->second) {
                // stale
            } else {
                // Check if backward has seen this state
                auto bit = best_g_b.find(top.state);
                if (bit != best_g_b.end()) {
                    // MEET!
                    out_moves = top.path;
                    for (int m : path_b[top.state]) out_moves.push_back(m);
                    return true;
                }

                if ((long long)best_g_f.size() < MAX_STATES_FWD) {
                    for (int m = 0; m < 14; ++m) {
                        State ns = apply_move(top.state, m);
                        int ng = top.g + 1;
                        auto nit = best_g_f.find(ns);
                        if (nit == best_g_f.end() || ng < nit->second) {
                            best_g_f[ns] = ng;
                            std::vector<int> np = top.path;
                            np.push_back(m);
                            path_f[ns] = np;
                            Node n;
                            n.g = ng;
                            n.f = ng + (int)(WEIGHT_FWD * heuristic(ns));
                            n.state = ns;
                            n.path = np;
                            open_f.push(n);
                        }
                    }
                }
            }
        }

        if (elapsed() > budget_seconds) return false;

        // ---- Expand backward ----
        if (!open_b.empty()) {
            Node top = open_b.top(); open_b.pop();
            out_expansions++;

            auto it = best_g_b.find(top.state);
            if (it != best_g_b.end() && top.g > it->second) {
                // stale
            } else {
                // Check if forward has seen this state
                auto fit = best_g_f.find(top.state);
                if (fit != best_g_f.end()) {
                    // MEET!
                    out_moves = path_f[top.state];
                    for (int m : top.path) out_moves.push_back(m);
                    return true;
                }

                if ((long long)best_g_b.size() < MAX_STATES_BWD) {
                    for (int m = 0; m < 14; ++m) {
                        State ns = apply_move(top.state, m);
                        int ng = top.g + 1;
                        auto nit = best_g_b.find(ns);
                        if (nit == best_g_b.end() || ng < nit->second) {
                            best_g_b[ns] = ng;
                            std::vector<int> np = top.path;
                            np.push_back(m);
                            path_b[ns] = np;
                            Node n;
                            n.g = ng;
                            n.f = ng + (int)(WEIGHT_BWD * heuristic(ns));
                            n.state = ns;
                            n.path = np;
                            open_b.push(n);
                        }
                    }
                }
            }
        }
    }

    return false;
}

// =============================================================================
// PARSER (same as before)
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

struct Problem { std::string cid; State initial; int total_len; };

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
    if (rels.size() < 2) throw std::runtime_error("need 2 relators");
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
        std::cerr << "Usage: solver_ac_bidir.exe <ac_unsolved.jsonl> <submission_bidir.txt>\n";
        return 1;
    }
    std::ios::sync_with_stdio(false);
    std::cout << std::unitbuf;

    std::string input_path  = argv[1];
    std::string output_path = argv[2];

    std::ifstream in(input_path);
    if (!in) { std::cerr << "cannot open " << input_path << "\n"; return 1; }

    std::vector<Problem> problems;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try { problems.push_back(parse_line(line)); } catch (...) {}
    }

    std::sort(problems.begin(), problems.end(),
              [](const Problem& a, const Problem& b){
                  return a.total_len < b.total_len;
              });

    std::cout << "Bidirectional A* solver\n";
    std::cout << "Loaded:             " << problems.size() << " problems\n";
    std::cout << "Step seconds:       " << STEP_SECONDS << "\n";
    std::cout << "Max per problem:    " << MAX_PER_PROBLEM << "s\n";
    std::cout << "Total run budget:   " << TOTAL_RUN_SECONDS << "s\n";
    std::cout << "Forward weight:     " << WEIGHT_FWD << "\n";
    std::cout << "Backward weight:    " << WEIGHT_BWD << "\n";
    std::cout << "Max fwd states:     " << MAX_STATES_FWD << "\n";
    std::cout << "Max bwd states:     " << MAX_STATES_BWD << "\n";
    std::cout << "Start:              " << timestamp() << "\n\n";

    std::ofstream out(output_path, std::ios::app);
    State target{{1}, {2}};

    auto overall_start = std::chrono::steady_clock::now();
    auto overall_elapsed = [&]() {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - overall_start).count();
    };

    int solved = 0, attempted = 0;

    for (size_t i = 0; i < problems.size(); ++i) {
        if (overall_elapsed() > TOTAL_RUN_SECONDS) {
            std::cout << "\n*** Overall budget reached. ***\n";
            break;
        }

        const Problem& p = problems[i];
        attempted++;

        bool found = false;
        std::vector<int> best_moves;
        int best_len = 999999;

        for (int budget = STEP_SECONDS; budget <= MAX_PER_PROBLEM; budget += STEP_SECONDS) {
            if (overall_elapsed() > TOTAL_RUN_SECONDS) break;

            std::vector<int> moves;
            long long exp = 0;
            bool ok = bidirectional_astar(p.initial, target, budget, moves, exp);

            if (ok) {
                if ((int)moves.size() < best_len) {
                    best_len = (int)moves.size();
                    best_moves = moves;
                    found = true;
                }
                break;
            }
        }

        std::cout << "[" << (i + 1) << "/" << problems.size() << "] "
                  << p.cid << " (len=" << p.total_len << ") ";

        if (found) {
            std::cout << "SOLVED (" << best_moves.size() << " moves)\n";
            out << p.cid << ": [";
            for (size_t k = 0; k < best_moves.size(); ++k) {
                if (k) out << ", ";
                out << best_moves[k];
            }
            out << "]\n";
            out.flush();
            solved++;
        } else {
            std::cout << "FAILED\n";
        }
    }

    std::cout << "\n==================================================\n";
    std::cout << "Done. Solved: " << solved << "/" << attempted << "\n";
    std::cout << "End: " << timestamp() << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "==================================================\n";

    return 0;
}