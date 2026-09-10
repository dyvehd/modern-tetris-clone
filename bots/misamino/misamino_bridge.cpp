// Linux bridge for the MisaMino search core (ai.cpp / genmove.cpp /
// tetris_gem.cpp — the portable part of the original Windows-only project,
// copied verbatim into this directory at build time). Replaces
// dllai.cpp/dllmain.cpp with a clean C ABI for ctypes.
//
// Convention mapping (verified against gamefield.h / ai.cpp):
//
//   MisaMino GameField: row[y] with y counted DOWNWARD, row[0] = top of a
//   23-row field, row[22] = the floor boundary (y+h > 22 collides), rows
//   [-20..-1] overflow above. Bit x of a row = column x — identical to
//   ours. Our 40-row field maps row[y] <-> our row 17 + y: the floor
//   row[22] = our bottom row 39, spawn gem_beg_y = 1 = our SPAWN_Y 18.
//   Gem (x, y, spin): x = bounding-box left column, y = bbox top row
//   (y-down); bitmap[h] bit j = bbox cell (row h, col j).
//
//   GEMTYPE ids: I=1 T=2 L=3 J=4 Z=5 S=6 O=7 (0 = none).
//   Our PieceType: I=0 J=1 L=2 O=3 S=4 T=5 Z=6.
//
// The bridge only reports placements as ABSOLUTE CELL SETS in our frame;
// the trainer re-navigates with our own movegen/pathfinder (the
// cheese-harness contract), so MisaMino's kick tables and input paths
// never run against our engine.

#include "ai.h"

#include <cstring>
#include <vector>

#define DLLEXPORT extern "C"

// ai.cpp defines these but ai.h does not declare them (the original DLL
// used them inside one translation unit).
namespace AI {
    int Evaluate(const GameField& field, int att, int clear, int depth, int player);
    int pasteClearAttack(GameField& field, int gemnum, int x, int y, int spin,
                         signed char wallkick_spin, int& att);
}

#define OUR_TOP_OF_FIELD 17   // our row index of mm row 0

static int from_mm_piece(int gemtype)
{
    static const int map[8] = {-1, 0, 5, 2, 1, 6, 4, 3};
    return map[gemtype];
}

struct MMResult {
    int ok;              // 1 = a placement was chosen
    int hold;            // 1 = hold first (placement is of the hold piece)
    int piece_id;        // our PieceType value of the placed piece
    int cells_r[4];      // absolute rows in our 40-row frame
    int cells_c[4];      // absolute columns
};

// One scored root candidate (the same enumeration the search branched on),
// exported so the trainer can rank the player's actual placement among
// them (chess-style annotation / live feedback).
struct MMCandidate {
    int hold;
    int piece_id;
    int cells_r[4];
    int cells_c[4];
    int score;           // the search's own root score for this placement
    int lines;           // lines this placement clears
};

static int cells_absolute(int gemnum, int x, int y, int spin,
                          int* out_r, int* out_c)
{
    const AI::Gem& g = AI::getGem(gemnum, spin);
    int n = 0;
    for (int h = 0; h < 4; ++h) {
        unsigned long bm = g.bitmap[h];
        if (!bm) continue;
        for (int j = 0; j < 4; ++j) {
            if (bm & (1UL << j)) {
                int r = OUR_TOP_OF_FIELD + y + h;
                int c = x + j;
                if (r < 0 || r > 39 || c < 0 || c > 9) return 0;
                out_r[n] = r;
                out_c[n] = c;
                ++n;
            }
        }
    }
    return n;
}

// Score one root candidate exactly the way the search's own root loop
// (ai.cpp AISearch) scores it: paste + clear + attack on a field copy,
// then the follow-up pieces' best line via a nested AISearch, with the
// attack/clear accumulation carried in (lastatt/lastclear).
static int score_candidate(const AI::GameField& field, int gemnum,
                           const AI::MovingSimple& m, bool is_hold,
                           int active_id, const std::vector<AI::Gem>& next,
                           bool canhold, int lookahead)
{
    AI::GameField f2 = field;
    if (is_hold)
        f2.m_hold = active_id;  // the swapped-out piece sits in hold after
    int att = 0;
    int clear = AI::pasteClearAttack(f2, gemnum, m.x, m.y, m.spin,
                                     m.wallkick_spin, att);
    if (lookahead > 0 && !next.empty()) {
        std::vector<AI::Gem> rest(next.begin() + 1, next.end());
        int score = 0;
        AI::AISearch(f2, next[0], canhold, AI::gem_beg_x, AI::gem_beg_y,
                     rest, canhold, 0, 2, 0, lookahead - 1, &score,
                     att, clear, 1);
        return score;
    }
    return AI::Evaluate(f2, att, clear, 0, 0);
}

static void fill_field(AI::GameField& f, const unsigned short* rows40)
{
    for (int y = -17; y <= 22; ++y)
        f.row[y] = rows40[OUR_TOP_OF_FIELD + y];
}

DLLEXPORT
int mm_version() { return 1; }

// One think call. Single-threaded (the core keeps globals: combo table,
// 180/allspin flags); the Python side serializes calls per process.
//
// rows40       : 40 row bitmasks (row 0 = top, bit x = column x)
// n_next       : upcoming pieces in next_ids (after the active piece)
// next_ids     : GEMTYPE ids of the upcoming pieces
// hold_id      : GEMTYPE id of the held piece, 0 = none
// can_hold     : hold enabled AND unused on this piece
// active_id    : GEMTYPE id of the active piece
// b2b, combo   : engine b2b chain / combo count (MisaMino semantics:
//                combo 1 after the first clear, b2b 1 after the first
//                difficult clear — same as ours)
// lookahead    : follow-up pieces scored per candidate (2 = the search's
//                own root scoring; 0 = one-ply Evaluate only)
// out_best     : the chosen placement (MMResult)
// cands        : caller-allocated array, capacity cap_cands
// out_n_cands  : candidates written, sorted best-first
//
// Returns 1 when a placement was chosen. With lookahead = 2 the top
// candidate is by construction the search's own choice (the candidate
// scoring mirrors ai.cpp's root loop exactly).
DLLEXPORT
int mm_think(const unsigned short* rows40,
             int n_next, const unsigned char* next_ids,
             int hold_id, int can_hold, int active_id,
             int b2b, int combo, int lookahead,
             MMResult* out_best,
             MMCandidate* cands, int cap_cands, int* out_n_cands)
{
    if (out_n_cands) *out_n_cands = 0;
    if (out_best) out_best->ok = 0;

    AI::GameField field(10, 22);
    fill_field(field, rows40);
    field.m_hold = hold_id;
    field.b2b = b2b;
    field.combo = combo;

    std::vector<AI::Gem> next;
    for (int i = 0; i < n_next; ++i)
        next.push_back(AI::getGem(next_ids[i], 0));

    AI::MovingSimple best = AI::AISearch(field, AI::getGem(active_id, 0),
                                         can_hold != 0, AI::gem_beg_x,
                                         AI::gem_beg_y, next, can_hold != 0,
                                         0, 2, 0);
    // MovingSimple's default ctor leaves y/spin/hold uninitialized; only
    // x is a reliable "no move" marker.
    if (best.x == AI::MovingSimple::INVALID_POS)
        return 0;

    struct Enum {
        AI::MovingSimple m;
        int gemnum;
        bool hold;
        int score;
        int lines;
    };
    std::vector<Enum> out;
    std::vector<AI::MovingSimple> movs;

    AI::GenMoving(field, movs, AI::getGem(active_id, 0), AI::gem_beg_x,
                  AI::gem_beg_y, false);
    for (size_t i = 0; i < movs.size(); ++i) {
        AI::GameField ftmp = field;
        int att = 0;
        int lines = AI::pasteClearAttack(ftmp, active_id, movs[i].x, movs[i].y,
                                         movs[i].spin, movs[i].wallkick_spin, att);
        out.push_back((Enum){movs[i], active_id, false,
                              score_candidate(field, active_id, movs[i], false,
                                              active_id, next, can_hold != 0,
                                              lookahead),
                              lines});
    }

    if (can_hold && (hold_id != 0 || n_next > 0)) {
        int hold_gem = hold_id != 0 ? hold_id : next_ids[0];
        std::vector<AI::Gem> rest = (hold_id != 0)
            ? next
            : std::vector<AI::Gem>(next.begin() + 1, next.end());
        AI::GenMoving(field, movs, AI::getGem(hold_gem, 0), AI::gem_beg_x,
                      AI::gem_beg_y, true);
        for (size_t i = 0; i < movs.size(); ++i) {
            AI::GameField ftmp = field;
            int att = 0;
            int lines = AI::pasteClearAttack(ftmp, hold_gem, movs[i].x, movs[i].y,
                                             movs[i].spin, movs[i].wallkick_spin, att);
            out.push_back((Enum){movs[i], hold_gem, true,
                                  score_candidate(field, hold_gem, movs[i], true,
                                                  active_id, rest, can_hold != 0,
                                                  lookahead),
                                  lines});
        }
    }

    // sort best-first (stable, insertion sort)
    for (size_t i = 1; i < out.size(); ++i) {
        Enum key = out[i];
        size_t j = i;
        while (j > 0 && out[j - 1].score < key.score) {
            out[j] = out[j - 1];
            --j;
        }
        out[j] = key;
    }

    int n_out = 0;
    int best_rank = -1;
    for (size_t i = 0; i < out.size(); ++i) {
        // the search's chosen candidate: same position + spin + arrival
        // + hold flag (MovingSimple::operator==)
        if (best_rank < 0 && out[i].hold == best.hold && out[i].m == best) {
            best_rank = (int)i;
            if (out_best) {
                out_best->ok = 1;
                out_best->hold = best.hold ? 1 : 0;
                out_best->piece_id = from_mm_piece(out[i].gemnum);
                if (cells_absolute(out[i].gemnum, out[i].m.x, out[i].m.y,
                                   out[i].m.spin, out_best->cells_r,
                                   out_best->cells_c) != 4)
                    out_best->ok = 0;
            }
        }
        if (n_out < cap_cands && cands != nullptr) {
            MMCandidate& c = cands[n_out];
            c.hold = out[i].hold ? 1 : 0;
            c.piece_id = from_mm_piece(out[i].gemnum);
            c.score = out[i].score;
            c.lines = out[i].lines;
            if (cells_absolute(out[i].gemnum, out[i].m.x, out[i].m.y,
                               out[i].m.spin, c.cells_r, c.cells_c) != 4)
                continue;  // outside our 40-row frame; unreachable in practice
            ++n_out;
        }
    }
    if (out_n_cands) *out_n_cands = n_out;
    return (out_best && out_best->ok) ? 1 : (best_rank >= 0 ? 1 : 0);
}

// Configure the core's globals. combo_table indexes MisaMino-style combo
// counts (1 = first clear of the chain); the Jstris table in that
// indexing is [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, ...].
DLLEXPORT
void mm_configure(int spin180, int allspin, const int* combo_table,
                  int combo_table_len)
{
    AI::setSpin180(spin180 != 0);
    AI::setAllSpin(allspin != 0);
    if (combo_table && combo_table_len > 0)
        AI::setComboList(std::vector<int>(combo_table, combo_table + combo_table_len));
}
