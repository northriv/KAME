/*
 * rc_layout_probe.cpp — ground-truth layout for KAME_RC_TRACE slot
 * attribution (handoff §13.2), plus an end-to-end check that the dual-keyed
 * view markers land in the payload Packet's history.
 *
 * Slot addresses in captures are the `this` of the recording smart pointer;
 * attributing one to "container + offset" is only as good as the offsets
 * assumed.  This prints them from the compiler instead.  Run it once per
 * toolchain/ABI before trusting any slot arithmetic in a capture — the
 * §12.4/n=2 attribution "CASInfo+16 == old_wrapper" was refuted this way
 * (old_wrapper is at +8; +16 is PacketWrapper::m_packet).
 *
 * Build (not a ctest; -fno-access-control reaches Node's private internals):
 *   c++ -DA_NO_P1TREE -DKAME_RC_TRACE -fno-access-control \
 *       -I kamestm/tests -I kamestm -I kamepoolalloc \
 *       -O1 -g -std=gnu++17 -Wno-invalid-offsetof \
 *       -include kamestm/tests/support_standalone.h \
 *       kamestm/tests/rc_layout_probe.cpp kamestm/tests/rc_trace.cpp \
 *       kamestm/tests/support_standalone.cpp kamestm/threadlocal.cpp \
 *       -o rc_layout_probe
 */
#include <cstdio>
#include <cstddef>
#include "transaction.h"
#include "transaction_impl.h"

extern "C" void kame_rc_dump(const void *obj);

class LongNode : public Transactional::Node<LongNode> {
public:
    struct Payload : public Transactional::Node<LongNode>::Payload {
        long m_x = 0;
    };
};

using N  = Transactional::Node<LongNode>;
using PW = N::PacketWrapper;
using PK = N::Packet;
using VW = scoped_atomic_view<PW>;
using CI = N::CASInfo;

int main() {
    printf("sizeof(local_shared_ptr<PK>)      = %zu\n", sizeof(local_shared_ptr<PK>));
    printf("sizeof(local_weak_ptr<N::Linkage>)= %zu\n", sizeof(local_weak_ptr<N::Linkage>));
    printf("sizeof(scoped_atomic_view<PW>)    = %zu\n", sizeof(VW));
    printf("sizeof(PacketWrapper)             = %zu\n", sizeof(PW));
    printf("sizeof(Packet)                    = %zu\n", sizeof(PK));
    printf("sizeof(CASInfo)                   = %zu\n", sizeof(CI));
    printf("PW::m_bundledBy      @ +%zu\n", offsetof(PW, m_bundledBy));
    printf("PW::m_packet         @ +%zu\n", offsetof(PW, m_packet));
    printf("PW::m_reverse_index  @ +%zu\n", offsetof(PW, m_reverse_index));
    printf("PW::m_bundle_serial  @ +%zu\n", offsetof(PW, m_bundle_serial));
    printf("CI::linkage          @ +%zu\n", offsetof(CI, linkage));
    printf("CI::old_wrapper      @ +%zu\n", offsetof(CI, old_wrapper));
    printf("CI::new_wrapper      @ +%zu\n", offsetof(CI, new_wrapper));

#ifdef KAME_RC_TRACE
    // Dual-keying end-to-end: a view marker on a wrapper must land in the
    // payload Packet's history too (VADOPT from the move-in ctor, then a
    // VMOVE hop carrying the source slot).
    local_shared_ptr<PK> pkt(new PK());
    const void *pkt_addr = pkt.get();
    local_shared_ptr<PW> w(new PW(pkt, 1));
    atomic_shared_ptr<PW> aspw;
    {
        VW v(aspw, std::move(w));
        VW v2(std::move(v));
        printf("\n--- kame_rc_dump(Packet %p): expect VADOPT+VMOVE below ---\n",
            pkt_addr);
        fflush(stdout);
        kame_rc_dump(pkt_addr);
    }
#endif
    return 0;
}
