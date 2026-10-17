// =============================================================================
// Weighted A* solver for Andrews-Curtis problems.
//
// Weighted A* uses f = g + w*h instead of f = g + h, where w >= 1.
// A larger weight makes the search greedier: it trusts the heuristic more,
// expands fewer states, and often finds solutions much faster — at the
// cost of possibly not being the absolute shortest.
//
// Compile:
//   clang++ -std=c++17 -O3 -march=native solver_wa.cpp -o solver_wa.exe
//
// Run:
//   solver_wa.exe ac.jsonl submission_wa.txt
//
// To change the weight, edit WEIGHT below and recompile.
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
// TUNABLE PARAMETERS
// =============================================================================
static const int   WEIGHT              = 3;     // 1 = plain A*, 3 = greedy, 5 = very greedy
static const int   PER_PROBLEM_SECONDS = 30;    // 30 s per problem
static const int   TOTAL_RUN_SECONDS   = 1800;  // 30 min total run
static const int   MAX_TOTAL_LEN       = 20;    // only problems with |r0|+|r1| <= 20
static const long long MAX_STATES      = 3'000'000;

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
// HEURISTIC
//
// Same as before: total relator length.  Under WEIGHTED A*, this is
// multiplied by WEIGHT when computing f-values.
// =============================================================================
int heuristic(const State& s) {
    return (int)(s.first.size() + s.second.size());
}

// =============================================================================
// WEIGHTED A*
//
// f = g + WEIGHT * h
//
// When WEIGHT = 1, this is plain A*.
// When WEIGHT > 1, the search is greedier — it heads toward the goal
// faster but may not find the shortest solution.
// =============================================================================
struct Node {
    int f, g;
    State state;
    std::vector<int> path;
};

struct CompareNode {
    bool operator()(const Node& a, const Node& b) const {
        return a.f > b.f;    // min-heap by f
    }
};

bool weighted_astar(const State& initial,
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

    std::priority_queue<Node, std::vector<Node>, CompareNode> open;
    std::unordered_map<State, int, StateHash> best_g;

    Node start;
    start.g = 0;
    start.f = WEIGHT * heuristic(initial);
    start.state = initial;
    open.push(std::move(start));
    best_g[initial] = 0;

    out_expansions = 0;

    while (!open.empty()) {

        if (elapsed() > budget_seconds) return false;

        // Heartbeat
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                now - last_heartbeat).count() >= 5) {
            std::cout << "      [heartbeat] open=" << open.size()
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

        for (int m = 0; m < 14; ++m) {
            State ns = apply_move(top.state, m);
            int ng = top.g + 1;

            auto nit = best_g.find(ns);
            if (nit == best_g.end() || ng < nit->second) {
                best_g[ns] = ng;
                Node n;
                n.g = ng;
                // ------------------ THE KEY LINE ------------------
                n.f = ng + WEIGHT * heuristic(ns);
                // --------------------------------------------------
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
        std::cerr << "Usage: solver_wa.exe <input.jsonl> <output.txt>\n";
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

    std::cout << "Weighted A* solver\n";
    std::cout << "Weight:             " << WEIGHT << "\n";
    std::cout << "Loaded:             " << problems.size() << " problems (len <= "
              << MAX_TOTAL_LEN << ")\n";
    std::cout << "Per-problem budget: " << PER_PROBLEM_SECONDS << "s\n";
    std::cout << "Total run budget:   " << TOTAL_RUN_SECONDS << "s\n";
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
            std::cout << "\n*** Overall time budget reached. ***\n";
            break;
        }

        const Problem& p = problems[i];
        attempted++;

        std::vector<int> moves;
        long long expansions = 0;
        auto t0 = std::chrono::steady_clock::now();
        bool ok = weighted_astar(p.initial, target, PER_PROBLEM_SECONDS,
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
    std::cout << "Done. Weight=" << WEIGHT
              << "  Solved: " << solved << "/" << attempted << "\n";
    std::cout << "End: " << timestamp() << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "==================================================\n";

    return 0;
}