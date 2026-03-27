/*
 * win_compat.h — Linux kernel API shim for the Windows userspace GPIB port.
 *
 * Targets MinGW-w64 / Clang on Windows (x86_64 and arm64), Windows 7+.
 * Uses Win32 SRWLOCK + CONDITION_VARIABLE + CreateThread instead of pthreads.
 * libusb-1.0 provides USB access (same as the macOS/Linux port).
 *
 * Do not include directly — include compat.h which dispatches here on _WIN32.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */
#ifndef _WIN_COMPAT_H_
#define _WIN_COMPAT_H_

#ifndef _WIN32_WINNT
#  define _WIN32_WINNT 0x0601   /* Windows 7 — required for SRWLOCK, CONDITION_VARIABLE */
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
/* Pull in MinGW's <unistd.h> early so its include guard fires before libusb/Qt
 * can include it later.  Then strip any Sleep compat macro it defined so that
 * <windows.h> wins with the authoritative WINAPI declaration. */
#ifdef __MINGW32__
#  include <unistd.h>
#  ifdef Sleep
#    undef Sleep
#  endif
#endif
#include <windows.h>

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>       /* struct timespec (C11) */
#include <stdbool.h>
#include <stddef.h>     /* offsetof */

#include <libusb-1.0/libusb.h>

/* =========================================================
 * 1. Basic Types and Annotation Tags
 * ========================================================= */
typedef uint8_t  u8;
typedef uint64_t dma_addr_t;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t  __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef int32_t  s32;
typedef uint16_t __le16;
typedef uint32_t __le32;

#define __iomem
#define __init
#define __exit
#define __user
#define __must_check
#define __packed       __attribute__((packed))

#define KERNEL_VERSION(a, b, c) (((a) << 16) + ((b) << 8) + (c))
#define LINUX_VERSION_CODE KERNEL_VERSION(6, 15, 0)

#define cpu_to_le16(x) (x)
#define le16_to_cpu(x) (x)
#define cpu_to_le32(x) (x)
#define le32_to_cpu(x) (x)

/* =========================================================
 * 2. Memory Allocation
 * ========================================================= */
#define GFP_KERNEL  0
#define GFP_ATOMIC  0

#define kmalloc(size, flags)          malloc(size)
#define kzalloc(size, flags)          calloc(1, (size))
#define kmalloc_array(n, size, flags) malloc((size_t)(n) * (size_t)(size))
#define kfree(ptr)                    free(ptr)

/* =========================================================
 * 3. Circular Doubly-Linked List
 * ========================================================= */
struct list_head { struct list_head *next, *prev; };

#define INIT_LIST_HEAD(ptr) do { (ptr)->next = (ptr); (ptr)->prev = (ptr); } while (0)

static inline void list_add(struct list_head *new_node, struct list_head *head) {
    new_node->next = head->next;
    new_node->prev = head;
    head->next->prev = new_node;
    head->next = new_node;
}

static inline void list_del(struct list_head *entry) {
    entry->prev->next = entry->next;
    entry->next->prev = entry->prev;
    entry->next = NULL;
    entry->prev = NULL;
}

#define list_empty(head) ((head)->next == (head))

#define list_for_each_entry(pos, head, member) \
    for (pos = container_of((head)->next, typeof(*pos), member); \
         &pos->member != (head); \
         pos = container_of(pos->member.next, typeof(*pos), member))

/* =========================================================
 * 4. Logging and Module Macros
 * ========================================================= */
#define KERN_ERR   "[ERR] "
#define KERN_INFO  "[INFO] "
#define KERN_DEBUG "[DBG] "
#define printk(fmt, ...) printf(fmt, ##__VA_ARGS__)

struct device { char name[64]; void *driver_data; };

#define dev_err(dev, fmt, ...)   fprintf(stderr, "gpib: " fmt, ##__VA_ARGS__)
#define dev_info(dev, fmt, ...)  printf("gpib: " fmt,          ##__VA_ARGS__)
#define dev_warn(dev, fmt, ...)  printf("gpib: warn: " fmt,    ##__VA_ARGS__)
#define dev_dbg(dev, fmt, ...)   do {} while (0)
#define pr_info(fmt, ...)        printf("gpib: " fmt,          ##__VA_ARGS__)
#define pr_err(fmt, ...)         fprintf(stderr, "gpib: " fmt, ##__VA_ARGS__)
#define pr_warn(fmt, ...)        printf("gpib: warn: " fmt,    ##__VA_ARGS__)
#define pr_debug(fmt, ...)       do {} while (0)

#define BUG() do { \
    fprintf(stderr, "BUG: %s:%d\n", __FILE__, __LINE__); \
    abort(); \
} while (0)
#define BUG_ON(cond) do { if (cond) BUG(); } while (0)

#define MODULE_AUTHOR(s)
#define MODULE_DESCRIPTION(s)
#define MODULE_LICENSE(s)
#define MODULE_VERSION(s)
#define MODULE_DEVICE_TABLE(type, name)
#define MODULE_ALIAS_CHARDEV_MAJOR(x)
#define EXPORT_SYMBOL(sym)
#define EXPORT_SYMBOL_GPL(sym)
#define THIS_MODULE  NULL
#define KBUILD_MODNAME "ni_usb_gpib"
#define module_init(fn) int  ni_module_init(void)  { return (fn)(); }
#define module_exit(fn) void ni_module_exit(void)  { (fn)(); }

/* =========================================================
 * 5. Atomics
 * =========================================================
 * Same dual C/C++ strategy as osx_compat.h.
 * __atomic_* builtins work on both x86_64 and arm64 with GCC/Clang.
 * ========================================================= */
#ifdef __cplusplus
typedef int atomic_t;
#  define ATOMIC_INIT(i) (i)
static inline int  atomic_read(const atomic_t *v)
    { return __atomic_load_n(v, __ATOMIC_SEQ_CST); }
static inline void atomic_set(atomic_t *v, int i)
    { __atomic_store_n(v, i, __ATOMIC_SEQ_CST); }
static inline void atomic_inc(atomic_t *v)
    { __atomic_fetch_add(v, 1, __ATOMIC_SEQ_CST); }
static inline void atomic_dec(atomic_t *v)
    { __atomic_fetch_sub(v, 1, __ATOMIC_SEQ_CST); }
static inline int  atomic_inc_and_test(atomic_t *v)
    { return __atomic_fetch_add(v, 1, __ATOMIC_SEQ_CST) == -1; }
static inline int  atomic_dec_and_test(atomic_t *v)
    { return __atomic_fetch_sub(v, 1, __ATOMIC_SEQ_CST) == 1; }
#else
#  include <stdatomic.h>
typedef atomic_int atomic_t;
#  define ATOMIC_INIT(i) (i)
static inline int  atomic_read(const atomic_t *v)
    { return atomic_load((atomic_t *)v); }
static inline void atomic_set(atomic_t *v, int i)
    { atomic_store(v, i); }
static inline void atomic_inc(atomic_t *v)
    { atomic_fetch_add(v, 1); }
static inline void atomic_dec(atomic_t *v)
    { atomic_fetch_sub(v, 1); }
static inline int  atomic_inc_and_test(atomic_t *v)
    { return atomic_fetch_add(v, 1) == -1; }
static inline int  atomic_dec_and_test(atomic_t *v)
    { return atomic_fetch_sub(v, 1) == 1; }
#endif

/* Compiler barrier — works on x86_64 and arm64 with GCC/Clang (MinGW) */
#define smp_mb__before_atomic() __asm__ __volatile__("" ::: "memory")
#define smp_mb__after_atomic()  __asm__ __volatile__("" ::: "memory")

#define gpib_interface_struct gpib_interface
#define gpib_board_struct     gpib_board

/* =========================================================
 * 6. pthreads emulation — SRWLOCK + CONDITION_VARIABLE + CreateThread
 * =========================================================
 * SRWLOCK chosen over CRITICAL_SECTION because it supports SRWLOCK_INIT
 * for static initialisation (required by DEFINE_MUTEX at file scope).
 * CONDITION_VARIABLE pairs with SleepConditionVariableSRW.
 * Both available since Windows Vista / Windows 7.
 * ========================================================= */

/* --- mutex --- */
typedef SRWLOCK pthread_mutex_t;
#define PTHREAD_MUTEX_INITIALIZER       SRWLOCK_INIT
static inline void _pmtx_init(pthread_mutex_t *m)    { InitializeSRWLock(m); }
static inline void _pmtx_lock(pthread_mutex_t *m)    { AcquireSRWLockExclusive(m); }
static inline void _pmtx_unlock(pthread_mutex_t *m)  { ReleaseSRWLockExclusive(m); }
static inline int  _pmtx_trylock(pthread_mutex_t *m) {
    return TryAcquireSRWLockExclusive(m) ? 0 : EBUSY;
}
#define pthread_mutex_init(m, a)    (_pmtx_init(m), 0)
#define pthread_mutex_lock(m)       (_pmtx_lock(m), 0)
#define pthread_mutex_unlock(m)     (_pmtx_unlock(m), 0)
#define pthread_mutex_trylock(m)    _pmtx_trylock(m)
#define pthread_mutex_destroy(m)    ((void)(m))   /* SRWLOCK needs no cleanup */

/* --- condvar --- */
typedef CONDITION_VARIABLE pthread_cond_t;
#define pthread_cond_init(c, a)   (InitializeConditionVariable(c), 0)
#define pthread_cond_signal(c)    (WakeConditionVariable(c), 0)
#define pthread_cond_broadcast(c) (WakeAllConditionVariable(c), 0)
#define pthread_cond_destroy(c)   ((void)(c))

static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    SleepConditionVariableSRW(c, m, INFINITE, 0);
    return 0;
}

/* ts is expressed as GetTickCount64-epoch ms (matching _get_jiffies below) */
static inline int pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m,
                                          const struct timespec *ts) {
    ULONGLONG now_ms = GetTickCount64();
    ULONGLONG exp_ms = (ULONGLONG)ts->tv_sec * 1000ULL
                     + (ULONGLONG)(ts->tv_nsec / 1000000L);
    DWORD wait_ms = (exp_ms > now_ms) ? (DWORD)(exp_ms - now_ms) : 0;
    if (!SleepConditionVariableSRW(c, m, wait_ms, 0))
        return (GetLastError() == ERROR_TIMEOUT) ? ETIMEDOUT : -1;
    return 0;
}

/* --- threads --- */
typedef HANDLE pthread_t;

typedef struct { void *(*fn)(void *); void *arg; } _win_thread_tramp;
static DWORD WINAPI _win_thread_entry(LPVOID p) {
    _win_thread_tramp t = *(_win_thread_tramp *)p;
    free(p);
    t.fn(t.arg);
    return 0;
}
static inline int pthread_create(pthread_t *t, void *attr,
                                  void *(*fn)(void *), void *arg) {
    (void)attr;
    _win_thread_tramp *tr = (_win_thread_tramp *)malloc(sizeof(*tr));
    if (!tr) return ENOMEM;
    tr->fn = fn; tr->arg = arg;
    *t = CreateThread(NULL, 0, _win_thread_entry, tr, 0, NULL);
    if (!*t) { free(tr); return ENOMEM; }
    return 0;
}
static inline int pthread_join(pthread_t t, void **retval) {
    (void)retval;
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
    return 0;
}

/* =========================================================
 * 7. Spinlock (SRWLOCK)
 * ========================================================= */
typedef SRWLOCK spinlock_t;
typedef unsigned long irqflags_t;

#define spin_lock_init(l)             InitializeSRWLock(l)
#define spin_lock_irqsave(l, f)       do { (void)(f); AcquireSRWLockExclusive(l); } while (0)
#define spin_unlock_irqrestore(l, f)  do { (void)(f); ReleaseSRWLockExclusive(l); } while (0)
#define spin_lock(l)                  AcquireSRWLockExclusive(l)
#define spin_unlock(l)                ReleaseSRWLockExclusive(l)

typedef int irqreturn_t;
#define IRQ_NONE    0
#define IRQ_HANDLED 1
#define PT_REGS_ARG

/* =========================================================
 * 8. Mutex
 * ========================================================= */
struct mutex { pthread_mutex_t m; };

static inline void mutex_init(struct mutex *mtx)    { InitializeSRWLock(&mtx->m); }
static inline void mutex_lock(struct mutex *mtx)    { AcquireSRWLockExclusive(&mtx->m); }
static inline void mutex_unlock(struct mutex *mtx)  { ReleaseSRWLockExclusive(&mtx->m); }
static inline void mutex_destroy(struct mutex *mtx) { (void)mtx; }
static inline int  mutex_trylock(struct mutex *mtx) {
    return TryAcquireSRWLockExclusive(&mtx->m) ? 1 : 0;
}

/* SRWLOCK_INIT = {0} — safe for file-scope static variables */
#define DEFINE_MUTEX(name) struct mutex name = { SRWLOCK_INIT }

/* =========================================================
 * 9. Wait Queue
 * ========================================================= */
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} wait_queue_head_t;

static inline void init_waitqueue_head(wait_queue_head_t *q) {
    InitializeSRWLock(&q->lock);
    InitializeConditionVariable(&q->cond);
}

#define wake_up_interruptible(q) do { \
    AcquireSRWLockExclusive(&(q)->lock); \
    WakeAllConditionVariable(&(q)->cond); \
    ReleaseSRWLockExclusive(&(q)->lock); \
} while (0)
#define wake_up(q) wake_up_interruptible(q)

/* =========================================================
 * 10. Completion
 * ========================================================= */
struct completion {
    unsigned int    done;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
};

static inline void init_completion(struct completion *c) {
    c->done = 0;
    InitializeSRWLock(&c->lock);
    InitializeConditionVariable(&c->cond);
}

static inline void complete(struct completion *c) {
    AcquireSRWLockExclusive(&c->lock);
    c->done++;
    WakeConditionVariable(&c->cond);
    ReleaseSRWLockExclusive(&c->lock);
}

static inline void wait_for_completion(struct completion *c) {
    AcquireSRWLockExclusive(&c->lock);
    while (c->done == 0)
        SleepConditionVariableSRW(&c->cond, &c->lock, INFINITE, 0);
    c->done--;
    ReleaseSRWLockExclusive(&c->lock);
}

static inline int wait_for_completion_interruptible(struct completion *c) {
    wait_for_completion(c); return 0;
}

static inline unsigned long wait_for_completion_timeout(struct completion *c,
                                                         unsigned long t_ms) {
    ULONGLONG deadline = GetTickCount64() + (ULONGLONG)t_ms;
    AcquireSRWLockExclusive(&c->lock);
    while (c->done == 0) {
        ULONGLONG now = GetTickCount64();
        if (now >= deadline) break;
        SleepConditionVariableSRW(&c->cond, &c->lock, (DWORD)(deadline - now), 0);
    }
    int ret = (c->done > 0) ? 1 : 0;
    if (ret) c->done--;
    ReleaseSRWLockExclusive(&c->lock);
    return (unsigned long)ret;
}

/* =========================================================
 * 10b. Semaphore
 * ========================================================= */
struct semaphore {
    pthread_mutex_t lock;
    pthread_cond_t  cond;
    int             count;
};

static inline void sema_init(struct semaphore *sem, int count) {
    InitializeSRWLock(&sem->lock);
    InitializeConditionVariable(&sem->cond);
    sem->count = count;
}
static inline void up(struct semaphore *sem) {
    AcquireSRWLockExclusive(&sem->lock);
    sem->count++;
    WakeConditionVariable(&sem->cond);
    ReleaseSRWLockExclusive(&sem->lock);
}
static inline void down(struct semaphore *sem) {
    AcquireSRWLockExclusive(&sem->lock);
    while (sem->count <= 0)
        SleepConditionVariableSRW(&sem->cond, &sem->lock, INFINITE, 0);
    sem->count--;
    ReleaseSRWLockExclusive(&sem->lock);
}
static inline int down_interruptible(struct semaphore *sem) {
    down(sem); return 0;
}

/* =========================================================
 * 11. Time, jiffies, sleep
 * ========================================================= */
#define HZ 1000

/* GetTickCount64: ms since system boot, monotonic */
static inline unsigned long _get_jiffies(void) {
    return (unsigned long)GetTickCount64();
}
#define jiffies _get_jiffies()

static inline unsigned long msecs_to_jiffies(unsigned int ms) { return (unsigned long)ms; }
static inline unsigned int  jiffies_to_msecs(unsigned long j)  { return (unsigned int)j; }
#define time_after(a, b) ((long)(b) - (long)(a) < 0)

/* usleep/msleep: Windows Sleep() takes ms; round microseconds up.
 * #undef first in case a MinGW compat header (e.g. pthreads-win32 or
 * an old <unistd.h> shim) already defined these as macros calling Sleep. */
#undef usleep
#undef usleep_range
#undef msleep
#undef mdelay
#define usleep(us)             Sleep((DWORD)(((ULONGLONG)(us) + 999ULL) / 1000ULL))
#define usleep_range(min, max) usleep(min)
#define msleep(ms)             Sleep((DWORD)(ms))
#define mdelay(ms)             Sleep((DWORD)(ms))
static inline unsigned long msleep_interruptible(unsigned int ms) {
    Sleep((DWORD)ms); return 0;
}


/* =========================================================
 * 12. Timer (same logic as osx_compat.h, Win32 threading)
 * ========================================================= */
#define COMPAT_TIMER_ARG_TYPE struct timer_list *

struct timer_list {
    void          (*function)(COMPAT_TIMER_ARG_TYPE);
    unsigned long   expires;
    volatile int    active;
    volatile int    cancelled;
    pthread_t       thread;
    int             thread_started;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
};

static void *_timer_thread_func(void *arg) {
    struct timer_list *t = (struct timer_list *)arg;
    AcquireSRWLockExclusive(&t->lock);
    while (t->active && !t->cancelled) {
        unsigned long now = _get_jiffies();
        if ((long)(t->expires - now) <= 0) {
            t->active = 0;
            ReleaseSRWLockExclusive(&t->lock);
            if (t->function) t->function(t);
            return NULL;
        }
        DWORD ms_left = (DWORD)((long)(t->expires - now));
        SleepConditionVariableSRW(&t->cond, &t->lock, ms_left, 0);
    }
    t->active = 0;
    ReleaseSRWLockExclusive(&t->lock);
    return NULL;
}

static inline void COMPAT_TIMER_SETUP(struct timer_list *timer,
                                       void (*func)(COMPAT_TIMER_ARG_TYPE),
                                       unsigned int flags) {
    (void)flags;
    timer->function       = func;
    timer->expires        = 0;
    timer->active         = 0;
    timer->cancelled      = 0;
    timer->thread_started = 0;
    InitializeSRWLock(&timer->lock);
    InitializeConditionVariable(&timer->cond);
}

static inline int mod_timer(struct timer_list *t, unsigned long expires) {
    AcquireSRWLockExclusive(&t->lock);
    int was_active = t->active;
    t->expires = expires; t->cancelled = 0; t->active = 1;
    ReleaseSRWLockExclusive(&t->lock);
    return was_active;
}

static inline int del_timer_sync(struct timer_list *t) {
    AcquireSRWLockExclusive(&t->lock);
    int was_active = t->active;
    t->active = 0; t->cancelled = 1;
    WakeConditionVariable(&t->cond);
    ReleaseSRWLockExclusive(&t->lock);
    if (t->thread_started) { pthread_join(t->thread, NULL); t->thread_started = 0; }
    return was_active;
}
#define COMPAT_DEL_TIMER_SYNC(t) del_timer_sync(t)

#define COMPAT_FROM_TIMER(var, callback_timer, timer_fieldname) \
    container_of((struct timer_list *)(callback_timer), typeof(*(var)), timer_fieldname)

/* =========================================================
 * 13. USB structures and helpers  (identical to osx_compat.h)
 * ========================================================= */
#define _USB_PIPE_BULK_OUT  0x0000u
#define _USB_PIPE_BULK_IN   0x0100u
#define _USB_PIPE_INT_IN    0x0200u

#define usb_sndbulkpipe(dev, ep) (_USB_PIPE_BULK_OUT | ((unsigned int)(ep) & 0x7fu))
#define usb_rcvbulkpipe(dev, ep) (_USB_PIPE_BULK_IN  | ((unsigned int)(ep) & 0x7fu))
#define usb_rcvintpipe(dev, ep)  (_USB_PIPE_INT_IN   | ((unsigned int)(ep) & 0x7fu))
#define usb_rcvctrlpipe(dev, ep) (0x8000u)
#define usb_sndctrlpipe(dev, ep) (0x8001u)

#define USB_DIR_IN          0x80u
#define USB_DIR_OUT         0x00u
#define USB_TYPE_VENDOR     (0x02u << 5)
#define USB_RECIP_DEVICE    0x00u
#define USB_RECIP_INTERFACE 0x01u

struct usb_bus { int busnum; };

struct usb_device {
    libusb_device_handle *handle;
    struct device         dev;
    struct usb_bus       *bus;
    int                   devnum;
    struct { uint16_t idVendor; uint16_t idProduct; } descriptor;
    struct usb_bus _bus_storage;
};

struct usb_interface {
    struct device     dev;
    struct usb_device *udev;
};

static inline uint16_t USBID_TO_CPU(uint16_t id) { return id; }
static inline struct usb_device *interface_to_usbdev(struct usb_interface *i) { return i->udev; }
static inline void  usb_set_intfdata(struct usb_interface *i, void *d) { i->dev.driver_data = d; }
static inline void *usb_get_intfdata(struct usb_interface *i)           { return i->dev.driver_data; }
static inline void  usb_get_dev(struct usb_device *d) { (void)d; }
static inline void  usb_put_dev(struct usb_device *d) { (void)d; }
static inline int   usb_reset_configuration(struct usb_device *d) { (void)d; return 0; }
static inline void  usb_make_path(struct usb_device *d, char *buf, size_t len) {
    snprintf(buf, len, "/dev/usb/%d/%d", d->bus ? d->bus->busnum : 0, d->devnum);
}

struct usb_device_id {
    uint16_t idVendor; uint16_t idProduct; int bInterfaceNumber;
};
#define USB_DEVICE(vendor, product) \
    .idVendor = (vendor), .idProduct = (product), .bInterfaceNumber = -1
#define USB_DEVICE_INTERFACE_NUMBER(vendor, product, intf) \
    .idVendor = (vendor), .idProduct = (product), .bInterfaceNumber = (intf)

struct urb {
    struct usb_device *dev;
    unsigned int       pipe;
    void              *transfer_buffer;
    int                transfer_buffer_length;
    int                actual_length;
    int                status;
    void              *context;
    void             (*complete)(struct urb *);
    int                interval;
    volatile int       cancelled;
    pthread_t          thread;
    int                thread_started;
    pthread_mutex_t    lock;
};

static inline struct urb *usb_alloc_urb(int iso_packets, int flags) {
    (void)iso_packets; (void)flags;
    struct urb *u = (struct urb *)calloc(1, sizeof(struct urb));
    if (u) InitializeSRWLock(&u->lock);
    return u;
}
static inline void usb_free_urb(struct urb *u) { free(u); }

static inline void usb_fill_bulk_urb(struct urb *urb, struct usb_device *dev,
    unsigned int pipe, void *buf, int len, void (*complete)(struct urb *), void *ctx) {
    urb->dev = dev; urb->pipe = pipe;
    urb->transfer_buffer = buf; urb->transfer_buffer_length = len;
    urb->complete = complete; urb->context = ctx;
}
static inline void usb_fill_int_urb(struct urb *urb, struct usb_device *dev,
    unsigned int pipe, void *buf, int len,
    void (*complete)(struct urb *), void *ctx, int interval) {
    urb->dev = dev; urb->pipe = pipe;
    urb->transfer_buffer = buf; urb->transfer_buffer_length = len;
    urb->complete = complete; urb->context = ctx; urb->interval = interval;
}

static void *_urb_thread_func(void *arg) {
    struct urb *urb = (struct urb *)arg;
    unsigned int pipe_type = urb->pipe & 0xff00u;
    uint8_t ep = (uint8_t)(urb->pipe & 0x7fu);
    if (pipe_type == _USB_PIPE_INT_IN) {
        ep |= LIBUSB_ENDPOINT_IN;
        while (!urb->cancelled) {
            int actual = 0;
            int r = libusb_interrupt_transfer(urb->dev->handle, ep,
                (unsigned char *)urb->transfer_buffer,
                urb->transfer_buffer_length, &actual,
                (unsigned int)(urb->interval > 0 ? urb->interval * 10 : 1000));
            if (urb->cancelled) break;
            if (r == LIBUSB_ERROR_TIMEOUT) continue;
            urb->actual_length = actual;
            urb->status = (r == 0) ? 0 : -EIO;
            if (urb->complete) urb->complete(urb);
            if (urb->cancelled) break;
        }
    } else {
        if (pipe_type == _USB_PIPE_BULK_IN) ep |= LIBUSB_ENDPOINT_IN;
        int actual = 0;
        int r = libusb_bulk_transfer(urb->dev->handle, ep,
            (unsigned char *)urb->transfer_buffer,
            urb->transfer_buffer_length, &actual, 10000);
        if (urb->cancelled) { urb->status = -ECONNRESET; }
        else { urb->actual_length = actual; urb->status = (r == 0) ? 0 : -EIO; }
        if (urb->complete && !urb->cancelled) urb->complete(urb);
    }
    return NULL;
}

static inline int usb_submit_urb(struct urb *urb, int flags) {
    (void)flags;
    if (!urb || !urb->dev || !urb->dev->handle) return -ENODEV;
    urb->cancelled = 0; urb->status = 0; urb->actual_length = 0;
    unsigned int pipe_type = urb->pipe & 0xff00u;
    if (pipe_type == _USB_PIPE_INT_IN) {
        int r = pthread_create(&urb->thread, NULL, _urb_thread_func, urb);
        if (r) return -ENOMEM;
        urb->thread_started = 1;
    } else {
        uint8_t ep = (uint8_t)(urb->pipe & 0x7fu);
        if (pipe_type == _USB_PIPE_BULK_IN) ep |= LIBUSB_ENDPOINT_IN;
        int actual = 0;
        int r = libusb_bulk_transfer(urb->dev->handle, ep,
            (unsigned char *)urb->transfer_buffer,
            urb->transfer_buffer_length, &actual, 10000);
        urb->actual_length = actual;
        urb->status = (r == 0) ? 0 : -EIO;
        if (urb->complete && !urb->cancelled) urb->complete(urb);
    }
    return 0;
}

static inline void usb_kill_urb(struct urb *urb) {
    if (!urb) return;
    urb->cancelled = 1;
    if (urb->thread_started) { pthread_join(urb->thread, NULL); urb->thread_started = 0; }
}

static inline int USB_CONTROL_MSG(struct usb_device *dev, unsigned int pipe,
    uint8_t request, uint8_t requesttype, uint16_t value, uint16_t index,
    void *data, uint16_t size, int timeout_ms) {
    (void)pipe;
    return libusb_control_transfer(dev->handle, requesttype, request,
        value, index, (unsigned char *)data, size, (unsigned int)timeout_ms);
}

typedef int pm_message_t;

struct usb_driver {
    const char             *name;
    int  (*probe)(struct usb_interface *, const struct usb_device_id *);
    void (*disconnect)(struct usb_interface *);
    int  (*suspend)(struct usb_interface *, pm_message_t);
    int  (*resume)(struct usb_interface *);
    const struct usb_device_id *id_table;
};

extern struct usb_driver     *g_ni_usb_driver;
extern struct gpib_interface *g_ni_gpib_interface;

static inline int  usb_register(struct usb_driver *d) { g_ni_usb_driver = d; return 0; }
static inline void usb_deregister(struct usb_driver *d) { (void)d; }

/* =========================================================
 * 14. task_struct stub
 * ========================================================= */
struct task_struct { int pid; };

/* =========================================================
 * 15. Utility Macros
 * ========================================================= */
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#undef min
#undef max
#define min(x, y) ((x) < (y) ? (x) : (y))
#define max(x, y) ((x) > (y) ? (x) : (y))

#define DUMP_PREFIX_NONE 0
static inline void print_hex_dump(const char *level, const char *prefix_str,
    int prefix_type, int rowsize, int groupsize,
    const void *buf, size_t len, bool ascii) {
    (void)level; (void)prefix_str; (void)prefix_type; (void)groupsize; (void)ascii;
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < len; i++) {
        printf("%02x ", p[i]);
        if ((i + 1) % (size_t)rowsize == 0) printf("\n");
    }
    if (len % (size_t)rowsize) printf("\n");
}

struct pci_dev { int unused; };

/* =========================================================
 * 16. errno codes missing on Windows / MinGW
 * ========================================================= */
#ifndef ERESTARTSYS
#define ERESTARTSYS 512
#endif
#ifndef ENOTCONN
#define ENOTCONN    107
#endif
#ifndef ECOMM
#define ECOMM        70
#endif
#ifndef ESHUTDOWN
#define ESHUTDOWN   108
#endif

#endif /* _WIN_COMPAT_H_ */
