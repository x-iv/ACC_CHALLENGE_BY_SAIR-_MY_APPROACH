// =============================================================================
// Andrews-Curtis A* solver
//
// Compile with:
//   g++ -O3 -march=native -std=c++17 -o solver.exe solver.cpp
//
// Run with:
//   solver.exe <input.jsonl> <output.txt>
//
// This program reads a JSONL file of AC problems, tries to solve each one
// using A* search with a length-based heuristic, and writes the solutions
// it finds to the output file in the competition submission format.
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

// =============================================================================
// TYPES
//
// A "Word" is a sequence of signed integers representing a word in the free
// group on generators x and y.  The encoding is:
//     1  = x
//    -1  = x^-1
//     2  = y
//    -2  = y^-1
//
// A "State" is a pair of Words — the two relators (r0, r1) of the presentation.
// =============================================================================
using Word = std::vector<int8_t>;
using State = std::pair<Word, Word>;

// Time budget per problem in seconds.  Change this to test with different
// values.  Larger = more problems solved, but slower overall.
static const int TIME_BUDGET_SECONDS = 120;

// =============================================================================
// FREE REDUCTION
//
// Scans a word left to right with a stack.  Whenever the incoming letter is
// the inverse of the top of the stack, pop instead of push.  This cancels
// adjacent inverse pairs like "x X" or "y Y".  The result is the freely
// reduced form of the word, which is unique.
//
// Example: "y X X y x Y Y" reduces to "X" because yY cancels and the trailing
// YY reduces.
// =============================================================================
Word free_reduce(const Word& w) {
    Word out;
    out.reserve(w.size());
    for (int8_t a : w) {
        if (!out.empty() && out.back() == -a) {
            out.pop_back();   // cancel: a followed by a^-1 (or vice versa)
        } else {
            out.push_back(a);
        }
    }
    return out;
}

// =============================================================================
// WORD INVERSION
//
// Returns the formal inverse of a word: reverse the sequence and negate every
// letter.  For example, (x, y, x^-1) inverts to (x, y^-1, x^-1).
// =============================================================================
Word invert_word(const Word& w) {
    Word out(w.rbegin(), w.rend());
    for (auto& a : out) a = -a;
    return out;
}

// =============================================================================
// MOVE TABLE (ac-r2-v1)
//
// The 14 atomic AC moves.  Given a state (r0, r1), apply one move and return
// the new state.  Every move result is freely reduced so states are canonical.
//
// Move IDs:
//   0: invert r0
//   1: invert r1
//   2: r0 <- r0 * r1
//   3: r0 <- r0 * r1^-1
//   4: r1 <- r1 * r0
//   5: r1 <- r1 * r0^-1
//   6..9: conjugate r0 by x, x^-1, y, y^-1
//  10..13: conjugate r1 by x, x^-1, y, y^-1
// =============================================================================

// Generators used by conjugation moves.  Index 0 -> x, 1 -> x^-1, etc.
static const int8_t GENS[4] = {1, -1, 2, -2};

State apply_move(const State& s, int m) {
    const Word& r0 = s.first;
    const Word& r1 = s.second;

    switch (m) {
        // --- Inversions ---
        case 0: return {invert_word(r0), r1};
        case 1: return {r0, invert_word(r1)};

        // --- Multiplications of r0 ---
        case 2: {
            Word x = r0;
            x.insert(x.end(), r1.begin(), r1.end());
            return {free_reduce(x), r1};
        }
        case 3: {
            Word inv = invert_word(r1);
            Word x = r0;
            x.insert(x.end(), inv.begin(), inv.end());
            return {free_reduce(x), r1};
        }

        // --- Multiplications of r1 ---
        case 4: {
            Word x = r1;
            x.insert(x.end(), r0.begin(), r0.end());
            return {r0, free_reduce(x)};
        }
        case 5: {
            Word inv = invert_word(r0);
            Word x = r1;
            x.insert(x.end(), inv.begin(), inv.end());
            return {r0, free_reduce(x)};
        }

        // --- Conjugations of r0 ---
        case 6: case 7: case 8: case 9: {
            int8_t g = GENS[m - 6];
            Word x;
            x.push_back(g);
            x.insert(x.end(), r0.begin(), r0.end());
            x.push_back(-g);
            return {free_reduce(x), r1};
        }

        // --- Conjugations of r1 ---
        case 10: case 11: case 12: case 13: {
            int8_t g = GENS[m - 10];
            Word x;
            x.push_back(g);
            x.insert(x.end(), r1.begin(), r1.end());
            x.push_back(-g);
            return {r0, free_reduce(x)};
        }
    }

    throw std::runtime_error("bad move id");
}

// =============================================================================
// STATE HASHING
//
// std::unordered_map and std::unordered_set need a hash function for our
// custom State type.  This mixes the bytes of r0 and r1 into a single size_t.
// A simple polynomial hash with a large prime multiplier — good enough and
// fast.
// =============================================================================
struct StateHash {
    size_t operator()(const State& s) const {
        size_t h = 1469598103934665603ULL;  // FNV offset basis

        // Mix in r0
        for (auto a : s.first) {
            h ^= (size_t)(a + 10);
            h *= 1099511628211ULL;          // FNV prime
        }
        h ^= 0xFFFFULL;                     // separator between the two words
        h *= 1099511628211ULL;

        // Mix in r1
        for (auto a : s.second) {
            h ^= (size_t)(a + 10);
            h *= 1099511628211ULL;
        }
        return h;
    }
};

// =============================================================================
// HEURISTIC
//
// A* uses h(state) = "estimated cost to reach the goal".  Our heuristic is
// simply the total length of the two relators.  The goal is (x, y) with
// total length 2.  Shorter relators means we're closer to the goal, so
// minimizing length drives the search.
//
// h is admissible in the sense that no sequence of moves can reduce total
// length faster than one letter per move, but this is a heuristic, not a
// strict lower bound — it's "good enough" to guide A* efficiently.
// =============================================================================
int heuristic(const State& s) {
    return (int)(s.first.size() + s.second.size());
}

// =============================================================================
// A* SEARCH
//
// Standard A* over the state graph:
//   - g(state) = number of moves taken to reach this state
//   - f(state) = g(state) + h(state)   (estimated total cost)
//
// Uses a priority queue ordered by f.  Visited map stores the best g found
// so far for each state, to avoid re-expanding.
//
// Returns true and fills out_moves if a solution is found within the time
// budget, otherwise returns false.
// =============================================================================
struct Node {
    int f;                      // priority: g + h
    int g;                      // moves taken so far
    State state;                // the presentation state
    std::vector<int> path;      // sequence of move IDs applied
};

// Priority queue comparator: smaller f is higher priority
struct CompareNode {
    bool operator()(const Node& a, const Node& b) const {
        return a.f > b.f;
    }
};

bool astar(const State& initial,
           const State& target,
           std::vector<int>& out_moves,
           long long& out_expansions) {

    // Record start time for the time budget check
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(now - t0).count();
    };

    // Priority queue of nodes to expand
    std::priority_queue<Node, std::vector<Node>, CompareNode> open;

    // Best g seen for each state
    std::unordered_map<State, int, StateHash> best_g;

    // Push the initial state
    Node start;
    start.g = 0;
    start.f = heuristic(initial);
    start.state = initial;
    open.push(start);
    best_g[initial] = 0;

    out_expansions = 0;

    while (!open.empty()) {

        // Time budget check
        if (elapsed() > TIME_BUDGET_SECONDS) {
            return false;
        }

        // Pop the best node
        Node top = open.top();
        open.pop();
        out_expansions++;

        // Goal test
        if (top.state == target) {
            out_moves = top.path;
            return true;
        }

        // Skip stale entries: if we've already reached this state with a
        // smaller g, this copy is useless.
        auto it = best_g.find(top.state);
        if (it != best_g.end() && top.g > it->second) {
            continue;
        }

        // Expand: try all 14 moves
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

    return false;  // ran out of nodes (shouldn't happen if the search space is finite)
}

// =============================================================================
// JSON PARSING
//
// The JSONL format has one problem per line:
//   {"challenge_id": "ac-00001", "description": "Presentation: <...>."}
//
// We use a light regex-free extraction of the ID and description, then parse
// the description into our (r0, r1) state.
// =============================================================================

// Map a single encoded character to its integer.  Encoding rules:
//   'x' -> 1, 'X' -> -1, 'y' -> 2, 'Y' -> -2
int8_t char_to_int(char c) {
    switch (c) {
        case 'x': return 1;
        case 'X': return -1;
        case 'y': return 2;
        case 'Y': return -2;
    }
    throw std::runtime_error(std::string("bad char in word: ") + c);
}

// Expand "x^-1" to "X" (uppercase), "y^-1" to "Y", and strip whitespace.
// After this, every letter is a single character and the word can be
// processed letter by letter.
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

// Parse a single word like "y^-1 x y" into a Word tuple.
Word parse_word(const std::string& s) {
    std::string e = encode(s);
    Word out;
    out.reserve(e.size());
    for (char c : e) out.push_back(char_to_int(c));
    return out;
}

// =============================================================================
// MAIN
//
// Usage: solver.exe <input.jsonl> <output.txt>
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: solver.exe <input.jsonl> <output.txt>\n";
        return 1;
    }
    std::string input_path = argv[1];
    std::string output_path = argv[2];

    std::ifstream in(input_path);
    std::ofstream out(output_path);
    if (!in) { std::cerr << "cannot open input: " << input_path << "\n"; return 1; }
    if (!out) { std::cerr << "cannot open output: " << output_path << "\n"; return 1; }

    // The goal state: (x, y) = ((1,), (2,))
    State target{{1}, {2}};

    std::string line;
    int line_no = 0;
    int solved = 0, failed = 0;
    auto total_start = std::chrono::steady_clock::now();

    while (std::getline(in, line)) {
        line_no++;
        if (line.empty()) continue;

        // --- Locate the challenge_id field ---
        size_t cid_pos = line.find("\"challenge_id\"");
        size_t desc_pos = line.find("\"description\"");
        if (cid_pos == std::string::npos || desc_pos == std::string::npos) continue;

        size_t cid_start = line.find('"', cid_pos + 14);
        size_t cid_end = line.find('"', cid_start + 1);
        std::string cid = line.substr(cid_start + 1, cid_end - cid_start - 1);

        // --- Locate the description text (between the opening and closing quotes) ---
        size_t desc_start = line.find('"', desc_pos + 13);
        size_t desc_end = line.rfind('"');
        std::string desc = line.substr(desc_start + 1, desc_end - desc_start - 1);

        // --- Parse "Presentation: <x, y | rel0 ; rel1>." ---
        size_t bar = desc.find('|');
        if (bar == std::string::npos) continue;
        std::string rels_part = desc.substr(bar + 1);
        size_t gt = rels_part.find('>');
        if (gt != std::string::npos) rels_part = rels_part.substr(0, gt);

        // --- Split relators by ';' and parse each ---
        std::vector<Word> rels;
        std::stringstream ss(rels_part);
        std::string chunk;
        while (std::getline(ss, chunk, ';')) {
            // Trim leading and trailing whitespace / dots
            while (!chunk.empty() && chunk.front() == ' ') chunk.erase(0, 1);
            while (!chunk.empty() && (chunk.back() == ' ' || chunk.back() == '.'))
                chunk.pop_back();
            // Remove "= 1" suffix
            size_t eq = chunk.find("= 1");
            if (eq != std::string::npos) chunk = chunk.substr(0, eq);
            while (!chunk.empty() && chunk.back() == ' ') chunk.pop_back();
            if (!chunk.empty()) rels.push_back(parse_word(chunk));
        }
        if (rels.size() < 2) continue;

        // --- Form the initial state ---
        State initial{rels[0], rels[1]};
        int L = (int)(initial.first.size() + initial.second.size());

        // --- Run A* ---
        std::vector<int> moves;
        long long expansions = 0;
        auto t0 = std::chrono::steady_clock::now();
        bool ok = astar(initial, target, moves, expansions);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // --- Print progress ---
        std::cout << "[" << line_no << "] " << cid
                  << " (len=" << L << ") ";
        if (ok) {
            std::cout << "SOLVED (" << moves.size() << " moves, "
                      << expansions << " exp, "
                      << elapsed << "s)\n";
            // Write to submission file
            out << cid << ": [";
            for (size_t i = 0; i < moves.size(); ++i) {
                if (i) out << ", ";
                out << moves[i];
            }
            out << "]\n";
            out.flush();
            solved++;
        } else {
            std::cout << "FAILED (" << expansions << " exp, "
                      << elapsed << "s)\n";
            failed++;
        }
    }

    auto total_end = std::chrono::steady_clock::now();
    double total_elapsed = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n";
    std::cout << "====================================\n";
    std::cout << "Solved:  " << solved << "\n";
    std::cout << "Failed:  " << failed << "\n";
    std::cout << "Total:   " << total_elapsed << "s\n";
    std::cout << "Output:  " << output_path << "\n";
    std::cout << "====================================\n";

    return 0;
}