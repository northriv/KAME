// TSD probe v4: try to reproduce the in-app garbage (raw slot at our key's
// offset non-null on worker threads) WITHOUT frameworks, by simulating what a
// Qt 6.8 app does before our ctor(101):
//   - key churn: many pthread_key_create (some with destructors, some deleted)
//   - thread churn: short-lived threads that SET those keys, then die
//     (exercises pthread-struct/stack recycling)
//   - then create OUR key (landing ~265 like the app), scan for its offset,
//     and inspect the raw slot on (a) a fresh pthread, (b) a GCD worker.
//
//   clang++ -std=gnu++17 -O2 tsd_probe.cpp -o /tmp/tsd_probe && /tmp/tsd_probe
//
#include <cstdio>
#include <cstdint>
#include <pthread.h>
#include <dispatch/dispatch.h>

static inline char *thread_pointer() {
#if defined(__aarch64__)
    uintptr_t tp; __asm__ volatile("mrs %0, TPIDRRO_EL0" : "=r"(tp)); return (char*)tp;
#elif defined(__x86_64__)
    uintptr_t tp; __asm__ volatile("movq %%gs:0, %0" : "=r"(tp)); return (char*)tp;
#else
    return nullptr;
#endif
}

enum { CHURN_KEYS = 120, CHURN_THREADS = 16 };
static pthread_key_t g_ck[CHURN_KEYS];
static pthread_key_t g_key;          // OUR key (like s_kame_page_key)
static long g_off = -1;
static volatile uintptr_t g_raw_pthread = ~0ull, g_raw_gcd = ~0ull;

static void dtor(void *) {}          // destructor presence changes cleanup paths

static void *churn_thread(void *arg) {
    // Store offset-like small values via the API (mimicking whatever the
    // frameworks keep in TSD), touching the recycled-struct hypothesis.
    for(int i = 0; i < CHURN_KEYS; ++i)
        pthread_setspecific(g_ck[i], (void*)(uintptr_t)(2000 + 8*i));
    return arg;
}

static void *fresh_worker(void *) {
    g_raw_pthread = *reinterpret_cast<uintptr_t*>(thread_pointer() + g_off);
    return nullptr;
}

int main() {
    // --- churn phase (before "our ctor") ---
    for(int i = 0; i < CHURN_KEYS; ++i)
        pthread_key_create(&g_ck[i], (i % 3) ? dtor : nullptr);
    for(int t = 0; t < CHURN_THREADS; ++t) {
        pthread_t th; pthread_create(&th, nullptr, churn_thread, nullptr);
        pthread_join(th, nullptr);   // dies -> struct/stack recycled
    }
    // delete a fraction (frameworks do delete keys)
    for(int i = 0; i < CHURN_KEYS; i += 4) pthread_key_delete(g_ck[i]);

    // --- "our ctor" ---
    char *tp = thread_pointer();
    pthread_key_create(&g_key, nullptr);
    std::printf("our key = %u\n", (unsigned)g_key);
    const uintptr_t sent = 0xDEAD600D11AA1234ull;
    pthread_setspecific(g_key, (void*)sent);
    for(std::size_t o = 0; o < 8192 && g_off < 0; o += 8)
        if(*reinterpret_cast<uintptr_t*>(tp + o) == sent) g_off = (long)o;
    std::printf("scan offset = %ld\n", g_off);
    if(g_off < 0) return 1;
    pthread_setspecific(g_key, (void*)0x4444555566667777ull);

    // (a) fresh pthread AFTER churn (recycled struct candidate)
    pthread_t th; pthread_create(&th, nullptr, fresh_worker, nullptr);
    pthread_join(th, nullptr);
    std::printf("(a) fresh pthread raw slot = 0x%llx  %s\n",
        (unsigned long long)g_raw_pthread,
        g_raw_pthread ? "NON-NULL  <== reproduced!" : "null (clean)");

    // (b) GCD worker (the app crashed on dispatch root-queue workers too)
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
        g_raw_gcd = *reinterpret_cast<uintptr_t*>(thread_pointer() + g_off);
        dispatch_semaphore_signal(sem);
    });
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    std::printf("(b) GCD worker raw slot   = 0x%llx  %s\n",
        (unsigned long long)g_raw_gcd,
        g_raw_gcd ? "NON-NULL  <== reproduced!" : "null (clean)");
    return 0;
}
