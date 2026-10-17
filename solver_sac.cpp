// =============================================================================
// Stable AC solver — length <= 14 problems, target = empty presentation.
//
// Compile:
//   clang++ -std=c++17 -O3 -march=native solver_sac.cpp -o solver_sac.exe
//
// Run:
//   solver_sac.exe sac_len14.jsonl submission_sac_len14.txt
//
// Budgets: 30s per problem, 1800s total.
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

// A Stable AC state is a list of relators. Each relator is a Word
// (vector of int8_t). The generators in play are implicit: any integer
// appearing in any relator is "in play".
using Word  = std::vector<int8_t>;
using State = std::vector<Word>;   // list of relators, no fixed size

// ------------- free reduction + inversion -------------
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

// ------------- stable AC moves -------------
// Move IDs 0..13: identical to AC, act on relators [0] and [1] (if present).
// Move 14: add a new generator g with relator (g,).
// Move 15: remove relator i if it is a single positive letter and that
//          letter appears nowhere else; also removes the generator.
// Everything above 15 is a combination move — we implement the essentials.
// -------------

static const int8_t GENS[4] = {1, -1, 2, -2};

// Apply one stable AC move. Returns false if the move is not applicable.
bool apply_sac_move(const State& in, int m, State& out) {
    out = in;

    // ---- Moves 0..13: AC moves on relators 0 and 1 ----
    if (m >= 0 && m <= 13) {
        if (out.size() < 2) return false;
        Word r0 = out[0], r1 = out[1];
        switch (m) {
            case 0:  r0 = invert_word(r0); break;
            case 1:  r1 = invert_word(r1); break;
            case 2:  { Word x = r0; x.insert(x.end(), r1.begin(), r1.end()); r0 = free_reduce(x); } break;
            case 3:  { Word inv = invert_word(r1); Word x = r0; x.insert(x.end(), inv.begin(), inv.end()); r0 = free_reduce(x); } break;
            case 4:  { Word x = r1; x.insert(x.end(), r0.begin(), r0.end()); r1 = free_reduce(x); } break;
            case 5:  { Word inv = invert_word(r0); Word x = r1; x.insert(x.end(), inv.begin(), inv.end()); r1 = free_reduce(x); } break;
            case 6:  { Word x; x.push_back(1);  x.insert(x.end(), r0.begin(), r0.end()); x.push_back(-1); r0 = free_reduce(x); } break;
            case 7:  { Word x; x.push_back(-1); x.insert(x.end(), r0.begin(), r0.end()); x.push_back(1);  r0 = free_reduce(x); } break;
            case 8:  { Word x; x.push_back(2);  x.insert(x.end(), r0.begin(), r0.end()); x.push_back(-2); r0 = free_reduce(x); } break;
            case 9:  { Word x; x.push_back(-2); x.insert(x.end(), r0.begin(), r0.end()); x.push_back(2);  r0 = free_reduce(x); } break;
            case 10: { Word x; x.push_back(1);  x.insert(x.end(), r1.begin(), r1.end()); x.push_back(-1); r1 = free_reduce(x); } break;
            case 11: { Word x; x.push_back(-1); x.insert(x.end(), r1.begin(), r1.end()); x.push_back(1);  r1 = free_reduce(x); } break;
            case 12: { Word x; x.push_back(2);  x.insert(x.end(), r1.begin(), r1.end()); x.push_back(-2); r1 = free_reduce(x); } break;
            case 13: { Word x; x.push_back(-2); x.insert(x.end(), r1.begin(), r1.end()); x.push_back(2);  r1 = free_reduce(x); } break;
        }
        out[0] = r0;
        out[1] = r1;
        // Drop empty relators
        if (out[0].empty()) out.erase(out.begin());
        else if (out.size() > 1 && out[1].empty()) out.erase(out.begin() + 1);
        return true;
    }

    // ---- Move 14: add a new generator g with relator (g,) ----
    if (m == 14) {
        if (out.size() >= 8) return false;   // max 8 generators
        // Find a fresh generator id
        std::unordered_set<int8_t> used;
        for (auto& r : out) for (auto a : r) used.insert((int8_t)std::abs(a));
        for (int8_t g = 1; g <= 8; ++g) {
            if (!used.count(g)) {
                out.push_back({g});
                return true;
            }
        }
        return false;
    }

    // ---- Move 15: remove a relator that is a single positive letter ----
    if (m == 15) {
        for (size_t i = 0; i < out.size(); ++i) {
            if (out[i].size() == 1 && out[i][0] > 0) {
                int8_t g = out[i][0];
                // Must not appear elsewhere
                bool appears_elsewhere = false;
                for (size_t j = 0; j < out.size(); ++j) {
                    if (j == i) continue;
                    for (auto a : out[j]) {
                        if (std::abs(a) == g) { appears_elsewhere = true; break; }
                    }
                    if (appears_elsewhere) break;
                }
                if (!appears_elsewhere) {
                    out.erase(out.begin() + i);
                    return true;
                }
            }
        }
        return false;
    }

    return false;
}

const int NUM_SAC_MOVES = 16;

// ------------- state hashing -------------
struct StateHash {
    size_t operator()(const State& s) const {
        size_t h = 1469598103934665603ULL;
        for (auto& r : s) {
            for (auto a : r) { h ^= (size_t)(a + 10); h *= 1099511628211ULL; }
            h ^= 0xFFULL; h *= 1099511628211ULL;
        }
        return h;
    }
};

// ------------- heuristic -------------
int heuristic_sac(const State& s) {
    int total = 0;
    for (auto& r : s) total += (int)r.size();
    return total + (int)s.size();   // length + count of relators
}

// ------------- A* -------------
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

bool astar_sac(const State& initial,
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
    start.f = heuristic_sac(initial);
    start.state = initial;
    open.push(std::move(start));
    best_g[initial] = 0;

    out_expansions = 0;

    while (!open.empty()) {

        if (elapsed() > budget_seconds) return false;

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(
                now - last_heartbeat).count() >= 5) {
            std::cout << "      [heartbeat] open=" << open.size()
                      << " seen=" << best_g.size()
                      << " elapsed=" << elapsed() << "s\n" << std::flush;
            last_heartbeat = now;
        }

        Node top = std::move(const_cast<Node&>(open.top()));
        open.pop();
        out_expansions++;

        if (top.state.empty()) {
            out_moves = std::move(top.path);
            return true;
        }

        auto it = best_g.find(top.state);
        if (it != best_g.end() && top.g > it->second) continue;

        for (int m = 0; m < NUM_SAC_MOVES; ++m) {
            State ns;
            if (!apply_sac_move(top.state, m, ns)) continue;
            int ng = top.g + 1;
            auto nit = best_g.find(ns);
            if (nit == best_g.end() || ng < nit->second) {
                best_g[ns] = ng;
                Node n;
                n.g = ng;
                n.f = ng + heuristic_sac(ns);
                n.state = ns;
                n.path = top.path;
                n.path.push_back(m);
                open.push(std::move(n));
            }
        }
    }

    return false;
}

// ------------- parser -------------
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
    if (rels.empty()) throw std::runtime_error("no relators");
    p.initial = rels;
    p.total_len = 0;
    for (auto& r : rels) p.total_len += (int)r.size();
    return p;
}

std::string timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

// ------------- main -------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: solver_sac.exe <input.jsonl> <output.txt>\n";
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
        try { problems.push_back(parse_line(line)); }
        catch (...) {}
    }

    std::sort(problems.begin(), problems.end(),
              [](const Problem& a, const Problem& b){
                  return a.total_len < b.total_len;
              });

    const int PER_PROBLEM_SECONDS = 120;
    const int TOTAL_RUN_SECONDS   = 1800;
    const long long MAX_STATES    = 6'000'000;

    std::cout << "Stable AC solver\n";
    std::cout << "Loaded:  " << problems.size() << " problems\n";
    std::cout << "Per-problem: " << PER_PROBLEM_SECONDS << "s\n";
    std::cout << "Total:       " << TOTAL_RUN_SECONDS << "s\n";
    std::cout << "Start: " << timestamp() << "\n\n";

    std::ofstream out(output_path, std::ios::app);

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
        bool ok = astar_sac(p.initial, PER_PROBLEM_SECONDS, moves, expansions);
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
    std::cout << "Done. Solved " << solved << "/" << attempted << "\n";
    std::cout << "End: " << timestamp() << "\n";
    std::cout << "Output: " << output_path << "\n";
    std::cout << "==================================================\n";

    return 0;
}