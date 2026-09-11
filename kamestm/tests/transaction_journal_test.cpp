/*
    Journal core: the bounded ring under contention, and the accounting that
    keeps the journal honest when it cannot keep up.

    What is asserted:
      1. Nothing is invented and nothing vanishes unaccounted: every record
         drained was captured, no record twice, and captured + dropped equals
         what was offered.
      2. A full ring refuses rather than blocks or overwrites, and counts what
         it refused.
      3. The ring keeps working around its wrap.
      4. isOlder() orders serials across the counter's wrap.
*/
#define msecsleep(x) (x)

#include "support_standalone.h"

#include <stdint.h>
#include <thread>
#include <vector>
#include <atomic>

#include "atomic_smart_ptr.h"
#include "transaction_journal.h"
#include "xthread.cpp"

using Transactional::Journal;

//A record big enough that it could not have been smuggled through a
//word-sized queue, as a real one would carry more than a value.
struct Rec {
    int32_t producer;
    int32_t index;
    int64_t payload;
};

enum : unsigned int {RING = 1024};
enum : int {PRODUCERS = 6, PER_PRODUCER = 20000};

static Journal<Rec, RING> s_journal;
static std::atomic<int> s_offered {0}, s_refused {0};

static void produce(int id) {
    for(int i = 0; i < PER_PRODUCER; ++i) {
        Rec r; r.producer = id; r.index = i; r.payload = (int64_t)id * 1000000 + i;
        s_offered.fetch_add(1);
        //serial: a per-thread counter in the upper bits and the thread id in
        //the low ones, as the framework's own serials are laid out.
        int64_t serial = ((int64_t)(i + 1) << 16) | id;
        if( !s_journal.capture(serial, r))
            s_refused.fetch_add(1);
    }
}

int main(int argc, char **argv) {
    //--- 1 & 2: many producers against a ring far too small for them
    std::vector<std::vector<char>> seen(PRODUCERS,
        std::vector<char>(PER_PRODUCER, 0));
    long long drained = 0, dropped_reported = 0;
    bool bad = false;

    std::atomic<bool> done {false};
    std::vector<std::thread> threads;
    for(int i = 0; i < PRODUCERS; ++i)
        threads.emplace_back(produce, i);

    auto sweep = [&]{
        dropped_reported += (long long)s_journal.takeDropped();
        drained += s_journal.drain([&](const Journal<Rec, RING>::Entry &e) {
            if((e.record.producer < 0) || (e.record.producer >= PRODUCERS) ||
               (e.record.index < 0) || (e.record.index >= PER_PRODUCER) ||
               (e.record.payload != (int64_t)e.record.producer * 1000000 + e.record.index) ||
               (e.serial != (((int64_t)(e.record.index + 1) << 16) | e.record.producer))) {
                printf("corrupt record: producer=%d index=%d payload=%lld serial=%lld\n",
                    (int)e.record.producer, (int)e.record.index,
                    (long long)e.record.payload, (long long)e.serial);
                bad = true;
                return;
            }
            char &mark = seen[e.record.producer][e.record.index];
            if(mark) {printf("duplicate record %d/%d\n", (int)e.record.producer, (int)e.record.index); bad = true;}
            mark = 1;
        });
    };
    while( !done) {
        sweep();
        bool alive = false;
        for(auto &&t: threads) (void)t, alive = true;
        if(s_offered.load() >= PRODUCERS * PER_PRODUCER) done = true;
        (void)alive;
    }
    for(auto &&t: threads) t.join();
    sweep();    //whatever the producers put in after the last sweep

    long long counted_seen = 0;
    for(auto &&v: seen) for(char c: v) counted_seen += c ? 1 : 0;

    printf("offered=%d  drained=%lld  dropped=%lld  distinct=%lld\n",
        s_offered.load(), drained, dropped_reported, counted_seen);
    if(bad) {printf("test1: failed (corrupt or duplicated records)\n"); return -1;}
    if(drained != counted_seen) {
        printf("test1: failed (drained %lld but only %lld distinct)\n", drained, counted_seen);
        return -1;
    }
    if(drained + dropped_reported != (long long)s_offered.load()) {
        printf("test1: failed (drained %lld + dropped %lld != offered %d)\n",
            drained, dropped_reported, s_offered.load());
        return -1;
    }
    if(dropped_reported != (long long)s_refused.load()) {
        printf("test1: failed (ring reported %lld drops, producers saw %d refusals)\n",
            dropped_reported, s_refused.load());
        return -1;
    }
    printf("test1 (nothing lost unaccounted, nothing duplicated): succeeded\n");

    //--- 3: a full ring refuses, and works again once drained
    {
        Journal<Rec, 8> jr;
        Rec r {}; int accepted = 0;
        for(int i = 0; i < 100; ++i) {
            r.index = i;
            if(jr.capture(i, r)) accepted++;
        }
        if((accepted != 8) || (jr.dropped() != 92)) {
            printf("test2: failed (accepted=%d dropped=%d)\n", accepted, (int)jr.dropped());
            return -1;
        }
        int got = jr.drain([](const Journal<Rec, 8>::Entry &){});
        if(got != 8) {printf("test2: failed (drained %d)\n", got); return -1;}
        //room again, and the count survives until taken
        if( !jr.capture(1000, r)) {printf("test2: failed (still refusing after drain)\n"); return -1;}
        if(jr.takeDropped() != 92) {printf("test2: failed (drop count not preserved)\n"); return -1;}
        if(jr.takeDropped() != 0) {printf("test2: failed (drop count not cleared)\n"); return -1;}
        printf("test2 (full ring refuses and recovers): succeeded\n");
    }

    //--- 4: many laps around the ring
    {
        Journal<Rec, 16> jr;
        Rec r {};
        for(int lap = 0; lap < 5000; ++lap) {
            r.index = lap;
            if( !jr.capture(lap, r)) {printf("test3: failed (refused at lap %d)\n", lap); return -1;}
            int n = jr.drain([&](const Journal<Rec, 16>::Entry &e) {
                if(e.record.index != lap) {printf("test3: failed (lap %d got %d)\n", lap, (int)e.record.index); }
            });
            if(n != 1) {printf("test3: failed (lap %d drained %d)\n", lap, n); return -1;}
        }
        printf("test3 (wraps cleanly): succeeded\n");
    }

    //--- 5: serial ordering across the counter's wrap
    {
        using J = Journal<Rec, 8>;
        int64_t big = INT64_MAX - 4, small = INT64_MIN + 4;   //wrapped past the end
        if( !J::isOlder(1, 2) || J::isOlder(2, 1) || J::isOlder(3, 3)) {
            printf("test4: failed (plain ordering)\n"); return -1;
        }
        if( !J::isOlder(big, small)) {   //big precedes small once wrapped
            printf("test4: failed (wrapped ordering)\n"); return -1;
        }
        printf("test4 (wrap-safe serial ordering): succeeded\n");
    }

    printf("all tests succeeded\n");
    return 0;
}
