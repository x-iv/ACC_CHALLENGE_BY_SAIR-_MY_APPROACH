// =============================================================================
// solver_simplified.cpp
//
// Simplified ACC solver with historical save and depth tracking.
//
// Features:
//   - Loads solved.jsonl  -> skips already-solved problems
//   - Loads failed.jsonl  -> starts from previous max_depth + 1
//   - Runs BFS_BOUNDED at increasing depths (1, 2, 3, ...)
//   - Saves every solve to solved.jsonl and submission.txt
//   - Saves every failure to failed.jsonl
//   - 30-minute global timeout
//
// Compile (Windows, clang):
//   clang++ -std=c++17 -O3 -march=native -D_CRT_SECURE_NO_WARNINGS \
//     solver_simplified.cpp -o solver_simplified.exe
//
// Run:
//   solver_simplified.exe ac_unsolved.jsonl
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
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

// =============================================================================
// CONFIG
// =============================================================================
static const int GLOBAL_TIMEOUT_SECONDS = 1800;   // 30 minutes
static const int MAX_DEPTH              = 12;
static const long long MAX_STATES_PER_DEPTH = 50'000'000;

static const std::string SOLVED_FILE     = "solved.jsonl";
static const std::string FAILED_FILE     = "failed.jsonl";
static const std::string SUBMISSION_FILE = "submission.txt";

// =============================================================================
// UTILITY: timestamp
//
// IMPORTANT: this must be defined BEFORE any function that uses it.
// That was the source of the original compile error.
// =============================================================================
std::string timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf;
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    return std::string(buf);
}

// =============================================================================
// CORE TYPES
// =============================================================================
using Word  = std::vector<int8_t>;
using State = std::pair<Word, Word>;

// =============================================================================
// WORD OPERATIONS
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

static const int8_t GENS[4] = {1, -1, 2, -2};

// =============================================================================
// AC MOVE
// =============================================================================
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
// STATE HASH
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
// PERSISTENCE — SolveRecord
// =============================================================================
struct SolveRecord {
    std::string cid;
    int depth = 0;
    std::vector<int> moves;
};

// Parse a JSON-ish line: {"challenge_id":"...", "depth":N, "moves":[...]}
// Hand-written parser (simple, no external deps).
SolveRecord parse_solved_line(const std::string& line) {
    SolveRecord rec;

    size_t p1 = line.find("\"challenge_id\":\"");
    if (p1 == std::string::npos) return rec;
    size_t p2 = line.find('"', p1 + 16);
    rec.cid = line.substr(p1 + 16, p2 - p1 - 16);

    size_t d1 = line.find("\"depth\":");
    if (d1 != std::string::npos) {
        rec.depth = std::stoi(line.substr(d1 + 8));
    }

    size_t m1 = line.find("\"moves\":[");
    if (m1 != std::string::npos) {
        size_t m2 = line.find(']', m1);
        std::string nums = line.substr(m1 + 9, m2 - m1 - 9);
        std::stringstream ss(nums);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) rec.moves.push_back(std::stoi(tok));
        }
    }
    return rec;
}

std::unordered_map<std::string, SolveRecord> load_solved() {
    std::unordered_map<std::string, SolveRecord> result;
    std::ifstream in(SOLVED_FILE);
    if (!in) return result;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        SolveRecord rec = parse_solved_line(line);
        if (!rec.cid.empty()) result[rec.cid] = rec;
    }
    return result;
}

std::unordered_map<std::string, int> load_failed() {
    std::unordered_map<std::string, int> result;
    std::ifstream in(FAILED_FILE);
    if (!in) return result;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t p1 = line.find("\"challenge_id\":\"");
        if (p1 == std::string::npos) continue;
        size_t p2 = line.find('"', p1 + 16);
        std::string cid = line.substr(p1 + 16, p2 - p1 - 16);

        size_t d1 = line.find("\"max_depth\":");
        if (d1 == std::string::npos) continue;
        int depth = std::stoi(line.substr(d1 + 12));

        result[cid] = depth;
    }
    return result;
}

void save_solve(const std::string& cid, int depth,
                const std::vector<int>& moves) {
    // Append to solved.jsonl
    {
        std::ofstream out(SOLVED_FILE, std::ios::app);
        out << "{\"challenge_id\":\"" << cid << "\","
            << "\"depth\":" << depth << ","
            << "\"moves\":[";
        for (size_t i = 0; i < moves.size(); ++i) {
            if (i) out << ",";
            out << moves[i];
        }
        out << "],"
            << "\"num_moves\":" << moves.size() << ","
            << "\"timestamp\":\"" << timestamp() << "\"}\n";
    }

    // Append to submission.txt (SAIR format)
    {
        std::ofstream sub(SUBMISSION_FILE, std::ios::app);
        sub << cid << ": [";
        for (size_t i = 0; i < moves.size(); ++i) {
            if (i) sub << ", ";
            sub << moves[i];
        }
        sub << "]\n";
    }

    std::cout << "  => SOLVED " << cid
              << " at depth=" << depth
              << " with " << moves.size() << " moves\n";
}

void save_failed(const std::string& cid, int max_depth) {
    std::ofstream out(FAILED_FILE, std::ios::app);
    out << "{\"challenge_id\":\"" << cid << "\","
        << "\"max_depth\":" << max_depth << ","
        << "\"timestamp\":\"" << timestamp() << "\"}\n";
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

struct Problem {
    std::string cid;
    State initial;
    int total_len = 0;
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

    if (rels.size() < 2) throw std::runtime_error("need 2 relators");
    p.initial = {rels[0], rels[1]};
    p.total_len = (int)(rels[0].size() + rels[1].size());
    return p;
}

// =============================================================================
// BFS_BOUNDED
// =============================================================================
struct BFSNode {
    State state;
    std::vector<int> path;
};

bool bfs_bounded(const State& initial,
                 const State& target,
                 int max_depth,
                 int time_limit_seconds,
                 std::vector<int>& out_moves) {

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - t0).count();
    };

    std::queue<BFSNode> q;
    std::unordered_set<State, StateHash> visited;

    q.push({initial, {}});
    visited.insert(initial);

    long long expansions = 0;

    while (!q.empty()) {
        if (elapsed() > time_limit_seconds) return false;
        if ((long long)visited.size() > MAX_STATES_PER_DEPTH) return false;

        BFSNode node = q.front();
        q.pop();

        if (node.state == target) {
            out_moves = node.path;
            return true;
        }

        if ((int)node.path.size() >= max_depth) continue;

        for (int m = 0; m < 14; ++m) {
            State ns = apply_move(node.state, m);
            if (visited.find(ns) == visited.end()) {
                visited.insert(ns);
                std::vector<int> np = node.path;
                np.push_back(m);
                q.push({ns, np});
            }
        }

        expansions++;
        if (expansions % 100000 == 0) {
            std::cout << "        [bfs] depth=" << node.path.size()
                      << " queue=" << q.size()
                      << " visited=" << visited.size()
                      << " elapsed=" << elapsed() << "s\n" << std::flush;
        }
    }
    return false;
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: solver_simplified.exe <ac_unsolved.jsonl>\n";
        return 1;
    }
    std::ios::sync_with_stdio(false);
    std::cout << std::unitbuf;

    std::string input_path = argv[1];

    // ---- Load persistence ----
    auto solved = load_solved();
    auto failed = load_failed();

    std::cout << "Simplified ACC solver\n";
    std::cout << "Previously solved: " << solved.size() << "\n";
    std::cout << "Previously failed: " << failed.size() << "\n";
    std::cout << "Global timeout:    " << GLOBAL_TIMEOUT_SECONDS << "s\n";
    std::cout << "Max depth:         " << MAX_DEPTH << "\n\n";

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
        catch (...) { /* skip malformed lines */ }
    }

    std::sort(problems.begin(), problems.end(),
              [](const Problem& a, const Problem& b) {
                  return a.total_len < b.total_len;
              });

    std::cout << "Loaded " << problems.size() << " problems\n\n";

    // ---- Main loop ----
    auto global_start = std::chrono::steady_clock::now();
    auto global_elapsed = [&]() {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - global_start).count();
    };

    const State target{{1}, {2}};
    int new_solves = 0;
    int attempted  = 0;
    int skipped    = 0;

    for (size_t i = 0; i < problems.size(); ++i) {
        if (global_elapsed() > GLOBAL_TIMEOUT_SECONDS) {
            std::cout << "\n*** Global timeout reached. ***\n";
            break;
        }

        const Problem& p = problems[i];

        // Skip already-solved problems
        if (solved.count(p.cid)) {
            skipped++;
            continue;
        }

        attempted++;
        std::cout << "[" << (i + 1) << "/" << problems.size() << "] "
                  << p.cid << " (len=" << p.total_len << ")\n";

        // Determine starting depth from previous failures
        int start_depth = 1;
        if (failed.count(p.cid)) {
            start_depth = failed[p.cid] + 1;
        }

        bool found = false;

        // Iterative deepening
        for (int depth = start_depth; depth <= MAX_DEPTH; ++depth) {
            if (global_elapsed() > GLOBAL_TIMEOUT_SECONDS) break;

            std::vector<int> moves;
            int per_call_timeout = 60;
            bool ok = bfs_bounded(p.initial, target, depth,
                                  per_call_timeout, moves);

            if (ok) {
                save_solve(p.cid, depth, moves);
                new_solves++;
                found = true;
                break;
            } else {
                failed[p.cid] = depth;
            }
        }

        if (!found) {
            save_failed(p.cid, MAX_DEPTH);
            std::cout << "  => FAILED after depth " << MAX_DEPTH << "\n";
        }
    }

    std::cout << "\n==================================================\n";
    std::cout << "New solves:  " << new_solves << "\n";
    std::cout << "Attempted:   " << attempted << "\n";
    std::cout << "Skipped:     " << skipped << " (already solved)\n";
    std::cout << "Elapsed:     " << global_elapsed() << "s\n";
    std::cout << "Solved file: " << SOLVED_FILE << "\n";
    std::cout << "Failed file: " << FAILED_FILE << "\n";
    std::cout << "Submission:  " << SUBMISSION_FILE << "\n";
    std::cout << "==================================================\n";

    return 0;
}