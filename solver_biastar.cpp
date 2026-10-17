// =============================================================================
// Fixed Bidirectional A* solver for Andrews-Curtis problems.
//
// Compile:
//   clang++ -std=c++17 -O3 -march=native solver_biastar_fixed.cpp -o solver_biastar_fixed.exe
//
// Run:
//   solver_biastar_fixed.exe ac.jsonl submission_bi.txt
//
// Budgets: 30s per problem, 1800s total run (30 min).
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

using Word  = std::vector<int8_t>;
using State = std::pair<Word, Word>;

// ---- budgets ----
static const int PER_PROBLEM_SECONDS = 30;    // 30 s per problem
static const int TOTAL_RUN_SECONDS   = 1800;  // 30 min total
static const int MAX_TOTAL_LEN       = 20;
static const int MAX_STATES          = 3'000'000;

// =============================================================================
// CORE
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

int heuristic(const State& s) {
    return (int)(s.first.size() + s.second.size());
}

// Verify a path by replaying it
bool verify_path(const State& initial, const State& target,
                 const std::vector<int>& path) {
    State cur = initial;
    for (int m : path) {
        if (m < 0 || m > 13) return false;
        cur = apply_move(cur, m);
    }
    return cur == target;
}

// =============================================================================
// BIDIRECTIONAL A*
// =============================================================================
static const int INVERSE_MOVE[14] = {0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12};

struct BiNode {
    int f, g;
    State state;
};

struct BiCompare {
    bool operator()(const BiNode& a, const BiNode& b) const {
        return a.f > b.f;
    }
};

struct Side {
    std::priority_queue<BiNode, std::vector<BiNode>, BiCompare> open;
    std::unordered_map<State, int, StateHash> best_g;
    std::unordered_map<State, std::vector<int>, StateHash> paths;
};

bool bidirectional_astar(const State& initial,
                         const State& target,
                         int budget_seconds,
                         std::vector<int>& out_moves,
                         long long& out_expansions) {

    auto t0 = std::chrono::steady_clock::now();
    auto last_heartbeat = t0;
    auto elapsed = [&]() {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    if (initial == target) { out_moves = {}; return true; }

    Side fwd, bwd;

    fwd.open.push({heuristic(initial), 0, initial});
    fwd.best_g[initial] = 0;
    fwd.paths[initial] = {};

    bwd.open.push({heuristic(target), 0, target});
    bwd.best_g[target] = 0;
    bwd.paths[target] = {};

    out_expansions = 0;

    while (!fwd.open.empty() && !bwd.open.empty()) {

        if (elapsed() > budget_seconds) return false;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                now - last_heartbeat).count() >= 5) {
            std::cout << "      [heartbeat] fwd_open=" << fwd.open.size()
                      << " bwd_open=" << bwd.open.size()
                      << " fwd_seen=" << fwd.best_g.size()
                      << " bwd_seen=" << bwd.best_g.size()
                      << " elapsed=" << elapsed() << "s\n" << std::flush;
            last_heartbeat = now;
        }

        if ((long long)fwd.best_g.size() + (long long)bwd.best_g.size() > MAX_STATES) {
            return false;
        }

        bool expand_forward = fwd.open.size() <= bwd.open.size();
        Side& side  = expand_forward ? fwd : bwd;
        Side& other = expand_forward ? bwd : fwd;

        BiNode top = side.open.top();
        side.open.pop();
        out_expansions++;

        auto it = side.best_g.find(top.state);
        if (it != side.best_g.end() && top.g > it->second) continue;

        for (int m = 0; m < 14; ++m) {

            int actual_move = expand_forward ? m : INVERSE_MOVE[m];
            State ns = apply_move(top.state, actual_move);
            int ng = top.g + 1;

            // Check for meeting
            auto oit = other.best_g.find(ns);
            if (oit != other.best_g.end()) {

                std::vector<int> fwd_path, bwd_path;

                if (expand_forward) {
                    // We're on forward side; other is backward.
                    // forward path: initial -> ns
                    // Build it from top.state
                    fwd_path = side.paths[top.state];
                    fwd_path.push_back(actual_move);
                    // backward path: ns -> target (already stored as other.paths[ns])
                    bwd_path = other.paths[ns];
                } else {
                    // We're on backward side; other is forward.
                    // forward path: initial -> ns (from other)
                    fwd_path = other.paths[ns];
                    // backward path: ns -> target.
                    // We reached ns from top.state via actual_move.
                    // So target -> top.state is side.paths[top.state].
                    // And target -> ns is side.paths[top.state] + [actual_move].
                    // To go ns -> target, reverse and invert.
                    std::vector<int> target_to_ns = side.paths[top.state];
                    target_to_ns.push_back(actual_move);
                    for (auto rit = target_to_ns.rbegin();
                         rit != target_to_ns.rend(); ++rit) {
                        bwd_path.push_back(INVERSE_MOVE[*rit]);
                    }
                }

                std::vector<int> candidate = fwd_path;
                candidate.insert(candidate.end(), bwd_path.begin(), bwd_path.end());

                // <<< KEY FIX: verify the candidate path before returning it
                if (verify_path(initial, target, candidate)) {
                    out_moves = candidate;
                    return true;
                }
                // Otherwise keep searching; this meeting was spurious.
            }

            auto nit = side.best_g.find(ns);
            if (nit == side.best_g.end() || ng < nit->second) {
                side.best_g[ns] = ng;
                std::vector<int> npath = side.paths[top.state];
                npath.push_back(actual_move);
                side.paths[ns] = npath;
                BiNode n;
                n.g = ng;
                n.f = ng + heuristic(ns);
                n.state = ns;
                side.open.push(n);
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
        std::cerr << "Usage: solver_biastar_fixed.exe <input.jsonl> <output.txt>\n";
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
        try {
            Problem p = parse_line(line);
            if (p.total_len <= MAX_TOTAL_LEN) problems.push_back(p);
        } catch (...) {}
    }

    std::sort(problems.begin(), problems.end(),
              [](const Problem& a, const Problem& b){
                  return a.total_len < b.total_len;
              });

    std::cout << "Loaded " << problems.size() << " problems.\n";
    std::cout << "Per-problem budget: " << PER_PROBLEM_SECONDS << "s\n";
    std::cout << "Total run budget:   " << TOTAL_RUN_SECONDS << "s\n";
    std::cout << "Start: " << timestamp() << "\n\n";

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

        std::vector<int> moves;
        long long expansions = 0;
        auto t0 = std::chrono::steady_clock::now();
        bool ok = bidirectional_astar(p.initial, target, PER_PROBLEM_SECONDS,
                                      moves, expansions);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "[" << (i + 1) << "/" << problems.size() << "] "
                  << p.cid << " (len=" << p.total_len << ") ";

        if (ok) {
            std::cout << "SOLVED (" << moves.size() << " moves, "
                      << expansions << " exp, " << elapsed << "s)\n";
            std::cout << "      moves = [";
            for (size_t k = 0; k < moves.size(); ++k) {
                if (k) std::cout << ", ";
                std::cout << moves[k];
            }
            std::cout << "]\n";

            out << p.cid << ": [";
            for (size_t k = 0; k < moves.size(); ++k) {
                if (k) out << ", ";
                out << moves[k];
            }
            out << "]\n";
            out.flush();
            solved++;
        } else {
            std::cout << "FAILED (" << expansions << " exp, " << elapsed << "s)\n";
        }
    }

    std::cout << "\n==================================================\n";
    std::cout << "Run complete. Solved: " << solved << "/" << attempted << "\n";
    std::cout << "End: " << timestamp() << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "==================================================\n";

    return 0;
}