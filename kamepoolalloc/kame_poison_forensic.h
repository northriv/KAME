/***************************************************************************
        Copyright (C) 2002-2026 Kentaro Kitagawa
                           kitag@issp.u-tokyo.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public
        License as published by the Free Software Foundation; either
        version 2 of the License, or (at your option) any later version.

        You should have received a copy of the GNU General
        Public License and a list of authors along with this program;
        see the files COPYING and AUTHORS.
 ***************************************************************************/
#ifndef KAME_POISON_FORENSIC_H_
#define KAME_POISON_FORENSIC_H_
/*
 * Forensic free-poison (debug only; see DYNNODE_UAF_HANDOFF.md §13.4).
 *
 * The plain poison (§3 of the handoff: fill freed pool blocks with
 * 0xBAADF00DBAADF00D) makes a stale access IDENTIFIABLE — a refcount read
 * >= 2^48 can only be freed memory.  This variant makes it ATTRIBUTABLE:
 * the first two 64-bit words of a freed block (exactly where an intrusive
 * refcnt / weak_refcnt would sit) carry an index into a ring of free
 * records, so the KAME_RC_TRACE tripwire that catches the stale operation
 * can immediately print WHO freed the block the stale reference targets —
 * thread, call chain, time, original pointer, size — regardless of how
 * long ago the free happened or how much event traffic has evicted the
 * tracer's rings since.
 *
 * Token layout (words 0 and 1 of the freed block; the rest of the block
 * keeps the plain 0xBAADF00DBAADF00D so pointer dereferences still fault
 * loudly):
 *
 *   bits 63..48   0xBAAD   tag; guarantees the value is >= 2^48, so the
 *                          existing tripwire threshold classifies it
 *   bits 47..16   free-record counter (full 32 bits: the ring index is
 *                          ctr % RING, and the stored ctr verifies the
 *                          record was not overwritten by a ring wrap)
 *   bits 15..0    0x8000   drift absorber: stale fetch_sub/fetch_add on
 *                          the word move only these bits (up to +/-32767
 *                          operations), so the index stays decodable and
 *                          the decoder reports the drift = how many stale
 *                          count operations already hit this block
 *
 * Enable by compiling the allocator with -DKAME_POISON_FORENSIC (the
 * production build never defines it; every addition to allocator.cpp is
 * inside the #ifdef).  The tracer (kamestm/tests/rc_trace.cpp) resolves
 * kame_poison_decode() via dlsym at runtime, so the test binary and the
 * allocator library need not agree on the flag.
 */
#include <cstdint>
#include <cstddef>

#define KAME_POISON_TAG        0xBAADull
#define KAME_POISON_PAD        0x8000u
#define KAME_POISON_PLAIN      0xBAADF00DBAADF00DULL

struct kame_freerec {
    void *ptr;                  /* the freed block (== the stale target) */
    std::size_t size;           /* pool usable size */
    unsigned long long tsc;     /* rdtsc / cntvct at free time */
    std::uint32_t ctr;          /* full counter; verifies ring slot */
    std::uint32_t tid;          /* small per-thread ordinal (1-based) */
    const void *ret[4];         /* call chain of the free (best effort) */
    std::uint32_t nret;
};

extern "C" {
/* Decode a poisoned word.  Returns 1 and fills *out when `word` carries
 * the forensic tag and its ring record is still live (not overwritten);
 * 0 otherwise.  Exported by the allocator only when built with
 * -DKAME_POISON_FORENSIC. */
int kame_poison_decode(unsigned long long word, struct kame_freerec *out);
}

#endif /*KAME_POISON_FORENSIC_H_*/
