// =============================================================================
// Tiered Andrews-Curtis A* solver
//
// Groups problems by total relator length and gives each tier a different
// per-problem time budget.  Live-writes submission file as it goes, so
// Ctrl+C or a crash never loses solved problems.
//
// Compile:
//   clang++ -std=c++17 -O3 -march=native solver_tiered.cpp -o solver_tiered.exe
//
// Run:
//   solver_tiered.exe ac.jsonl submission.txt
//
// Each tier prints its start/finish time and running solved/total counts.
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

// =============================================================================
// TIME BUDGET PER TIER (seconds)
//
// tier_by_length(L) returns the budget for a problem whose total relator
// length is L.  Edit these constants to change per-tier behaviour.
// =============================================================================
struct Tier {
    int len_min;
    int len_max;
    int budget_seconds;
    const char* name;
};

static const Tier TIERS[] = {
    {  2, 14,   600, "T1 len 2-14  (10 min)" },
    { 15, 16,   600, "T2 len 15-16 (10 min)" },
    { 17, 20,  1200, "T3 len 17-20 (20 min)" },
    { 21, 25,  1800, "T4 len 21-25 (30 min)" },
    { 26, 30,  2400, "T5 len 26-30 (40 min)" },
    { 31, 200, 3600, "T6 len 31+   (60 min)" },
};
static const int NUM_TIERS = sizeof(TIERS) / sizeof(TIERS[0]);

int tier_index_for_length(int L) {
    for (int i = 0; i < NUM_TIERS; ++i) {
        if (L >= TIERS[i].len_min && L <= TIERS[i].len_max) return i;
    }
    return NUM_TIERS - 1;
}

// =============================================================================
// FREE REDUCTION  (stack cancellation)
// =============================================================================
Word free_reduce(const Word& w) {
    Word out;
    out.reserve(w.size());
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

// =============================================================================
// MOVE TABLE (ac-r2-v1)
// =============================================================================
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

// =============================================================================
// STATE HASHING  (FNV-style mix)
// =============================================================================
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
// HEURISTIC  (total relator length)
// =============================================================================
int heuristic(const State& s) {
    return (int)(s.first.size() + s.second.size());
}

// =============================================================================
// A* SEARCH  (parameterized time budget)
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

bool astar(const State& initial,
           const State& target,
           int time_budget_seconds,
           std::vector<int>& out_moves,
           long long& out_expansions) {

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(now - t0).count();
    };

    std::priority_queue<Node, std::vector<Node>, CompareNode> open;
    std::unordered_map<State, int, StateHash> best_g;

    Node start;
    start.g = 0;
    start.f = heuristic(initial);
    start.state = initial;
    open.push(start);
    best_g[initial] = 0;
    out_expansions = 0;

    while (!open.empty()) {
        if (elapsed() > time_budget_seconds) return false;

        Node top = open.top(); open.pop();
        out_expansions++;

        if (top.state == target) {
            out_moves = top.path;
            return true;
        }

        auto it = best_g.find(top.state);
        if (it != best_g.end() && top.g > it->second) continue;

        for (int m = 0; m < 14; ++m) {
            State ns = apply_move(top.state, m);
            int ng = top.g + 1;
            auto nit = best_g.find(ns);
            if (nit == best_g.end() || ng < nit->second) {
                best_g[ns] = ng;
                Node n;
                n.g = ng;
                n.f = ng + heuristic(ns);
                n.state = ns;
                n.path = top.path;
                n.path.push_back(m);
                open.push(n);
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
            out.push_back(toupper(w[i]));
            i += 3;
        } else if (w[i] != ' ') {
            out.push_back(w[i]);
        }
    }
    return out;
}

Word parse_word(const std::string& s) {
    std::string e = encode(s);
    Word out;
    out.reserve(e.size());
    for (char c : e) out.push_back(char_to_int(c));
    return out;
}

// =============================================================================
// PROBLEM RECORD
// =============================================================================
struct Problem {
    std::string cid;
    State initial;
    int total_len;
};

Problem parse_line(const std::string& line) {
    Problem p;
    size_t cid_pos  = line.find("\"challenge_id\"");
    size_t desc_pos = line.find("\"description\"");
    if (cid_pos == std::string::npos || desc_pos == std::string::npos) {
        throw std::runtime_error("missing field");
    }
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

// =============================================================================
// HELPERS
// =============================================================================
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
        std::cerr << "Usage: solver_tiered.exe <input.jsonl> <output.txt>\n";
        return 1;
    }
    std::string input_path  = argv[1];
    std::string output_path = argv[2];

    // ---- Load all problems ----
    std::ifstream in(input_path);
    if (!in) { std::cerr << "cannot open " << input_path << "\n"; return 1; }

    std::vector<Problem> problems;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        try { problems.push_back(parse_line(line)); }
        catch (...) { /* skip malformed */ }
    }
    std::cout << "Loaded " << problems.size() << " problems.\n";

    // ---- Bucket by tier ----
    std::vector<std::vector<int>> by_tier(NUM_TIERS);
    for (size_t i = 0; i < problems.size(); ++i) {
        int t = tier_index_for_length(problems[i].total_len);
        by_tier[t].push_back((int)i);
    }
    for (int t = 0; t < NUM_TIERS; ++t) {
        std::cout << "  " << TIERS[t].name << ": "
                  << by_tier[t].size() << " problems\n";
    }
    std::cout << "\n";

    // ---- Open submission file (append mode so nothing is lost on re-run) ----
    std::ofstream out(output_path, std::ios::app);

    // ---- Goal ----
    State target{{1}, {2}};

    int total_solved = 0;
    auto overall_start = std::chrono::steady_clock::now();

    // ---- Process each tier in order ----
    for (int t = 0; t < NUM_TIERS; ++t) {
        if (by_tier[t].empty()) continue;

        int budget = TIERS[t].budget_seconds;
        std::cout << "==================================================\n";
        std::cout << "TIER " << t << ": " << TIERS[t].name
                  << "  (" << by_tier[t].size() << " problems, "
                  << budget << "s each)\n";
        std::cout << "Start: " << timestamp() << "\n";
        std::cout << "==================================================\n";

        int tier_solved = 0;

        for (size_t k = 0; k < by_tier[t].size(); ++k) {
            int idx = by_tier[t][k];
            const Problem& p = problems[idx];

            std::vector<int> moves;
            long long expansions = 0;
            auto t0 = std::chrono::steady_clock::now();
            bool ok = astar(p.initial, target, budget, moves, expansions);
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();

            std::cout << "[" << (k + 1) << "/" << by_tier[t].size()
                      << "] " << p.cid
                      << " (len=" << p.total_len << ") ";
            if (ok) {
                std::cout << "SOLVED (" << moves.size() << " moves, "
                          << expansions << " exp, "
                          << elapsed << "s)\n";
                out << p.cid << ": [";
                for (size_t i = 0; i < moves.size(); ++i) {
                    if (i) out << ", ";
                    out << moves[i];
                }
                out << "]\n";
                out.flush();          // << live-write: submission.txt stays safe
                tier_solved++;
                total_solved++;
            } else {
                std::cout << "FAILED (" << expansions << " exp, "
                          << elapsed << "s)\n";
            }
        }

        std::cout << "\nTier " << t << " done: " << tier_solved << "/"
                  << by_tier[t].size() << " solved.  Time: " << timestamp()
                  << "\n\n";
    }

    auto overall_end = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(overall_end - overall_start).count();

    std::cout << "==================================================\n";
    std::cout << "ALL DONE\n";
    std::cout << "Total solved: " << total_solved << "/" << problems.size() << "\n";
    std::cout << "Total time:   " << total_elapsed << "s\n";
    std::cout << "Output file:  " << output_path << "\n";
    std::cout << "==================================================\n";

    return 0;
}