/*
 * osx_compat.h — Linux kernel API shim for the macOS userspace GPIB port.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */
#ifndef _MACOS_COMPAT_H_
#define _MACOS_COMPAT_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdbool.h>
#include <libusb-1.0/libusb.h>
/* semaphore.h / sem_init is deprecated on macOS; use pthread condvar instead */

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

#define __iomem        /* memory-mapped I/O tag, empty in userspace */
#define __init         /* kernel init section */
#define __exit         /* kernel exit section */
#define __user         /* user-space pointer tag */
#define __must_check
#define __packed       __attribute__((packed))

/* Pretend to be a modern kernel so all version guards take the "current" path */
#define KERNEL_VERSION(a, b, c) (((a) << 16) + ((b) << 8) + (c))
#define LINUX_VERSION_CODE KERNEL_VERSION(6, 15, 0)

/* Endian helpers (assuming LE host — Intel/Apple Silicon) */
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
#define WARN_ON(cond) ((void)(cond))

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
/* Expose the driver's init/exit as externally-visible symbols */
#define module_init(fn) int  ni_module_init(void)  { return (fn)(); }
#define module_exit(fn) void ni_module_exit(void)  { (fn)(); }

/* =========================================================
 * 5. Atomics
 * =========================================================
 * Clang's <stdatomic.h> defines atomic_load/store/exchange/… as macros.
 * GCC 13 libstdc++ <atomic> declares free functions with the same names.
 * When both are visible in a C++ TU the names clash (clang + libstdc++ mix).
 *
 * In C mode we use <stdatomic.h> as before.  In C++ mode we avoid including
 * any atomic header here (osx_compat.h is often pulled in via an extern "C"
 * block, which would prohibit C++ templates) and instead rely on the
 * compiler-builtin __atomic_* intrinsics, which need no header.  The
 * _Atomic qualifier on the typedef is a Clang/GCC C++ extension. */
#ifdef __cplusplus
/* Use plain int as storage; atomicity is provided by the __atomic_* builtins.
 * _Atomic int is rejected as argument to __atomic_* in C++ mode by clang. */
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

/* smp_mb__before/after_atomic: memory barriers around atomic ops.
 * In userspace on x86/x86-64 these are no-ops. */
#define smp_mb__before_atomic() __asm__ __volatile__("" ::: "memory")
#define smp_mb__after_atomic()  __asm__ __volatile__("" ::: "memory")

/* Upstream linux-gpib renamed struct tags in gpib_types.h; bridge to the
 * old names used by this userspace port. */
#define gpib_interface_struct gpib_interface
#define gpib_board_struct     gpib_board

/* =========================================================
 * 6. Spinlock (mapped to pthread mutex)
 * ========================================================= */
typedef pthread_mutex_t spinlock_t;
typedef unsigned long   irqflags_t;

#define spin_lock_init(l)             pthread_mutex_init(l, NULL)
#define spin_lock_irqsave(l, f)       do { (void)(f); pthread_mutex_lock(l); } while (0)
#define spin_unlock_irqrestore(l, f)  do { (void)(f); pthread_mutex_unlock(l); } while (0)
#define spin_lock(l)                  pthread_mutex_lock(l)
#define spin_unlock(l)                pthread_mutex_unlock(l)

typedef int irqreturn_t;
#define IRQ_NONE    0
#define IRQ_HANDLED 1
/* PT_REGS_ARG: empty on kernels >= 2.6.19 (our LINUX_VERSION_CODE is 6.15) */
#define PT_REGS_ARG

/* =========================================================
 * 7. Mutex
 * ========================================================= */
struct mutex { pthread_mutex_t m; };

static inline void mutex_init(struct mutex *mtx)    { pthread_mutex_init(&mtx->m, NULL); }
static inline void mutex_lock(struct mutex *mtx)    { pthread_mutex_lock(&mtx->m); }
static inline void mutex_unlock(struct mutex *mtx)  { pthread_mutex_unlock(&mtx->m); }
static inline void mutex_destroy(struct mutex *mtx) { pthread_mutex_destroy(&mtx->m); }
static inline int  mutex_trylock(struct mutex *mtx) { return pthread_mutex_trylock(&mtx->m) == 0 ? 1 : 0; }

#define DEFINE_MUTEX(name) struct mutex name = { .m = PTHREAD_MUTEX_INITIALIZER }

/* =========================================================
 * 8. Wait Queue (pthread cond-based)
 * ========================================================= */
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} wait_queue_head_t;

static inline void init_waitqueue_head(wait_queue_head_t *q) {
    pthread_mutex_init(&q->lock, NULL);
    pthread_cond_init(&q->cond, NULL);
}

#define wake_up_interruptible(q) do { \
    pthread_mutex_lock(&(q)->lock); \
    pthread_cond_broadcast(&(q)->cond); \
    pthread_mutex_unlock(&(q)->lock); \
} while (0)
#define wake_up(q) wake_up_interruptible(q)

/* =========================================================
 * 9. Completion
 * ========================================================= */
struct completion {
    unsigned int    done;
    pthread_mutex_t lock;
    pthread_cond_t  cond;
};

static inline void init_completion(struct completion *c) {
    c->done = 0;
    pthread_mutex_init(&c->lock, NULL);
    pthread_cond_init(&c->cond, NULL);
}

static inline void complete(struct completion *c) {
    pthread_mutex_lock(&c->lock);
    c->done++;
    pthread_cond_signal(&c->cond);
    pthread_mutex_unlock(&c->lock);
}

static inline void wait_for_completion(struct completion *c) {
    pthread_mutex_lock(&c->lock);
    while (c->done == 0)
        pthread_cond_wait(&c->cond, &c->lock);
    c->done--;
    pthread_mutex_unlock(&c->lock);
}

static inline int wait_for_completion_interruptible(struct completion *c) {
    wait_for_completion(c);
    return 0;
}

static inline unsigned long wait_for_completion_timeout(struct completion *c, unsigned long t_ms) {
    struct timespec ts;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    ts.tv_sec  = tv.tv_sec + (long)(t_ms) / 1000;
    ts.tv_nsec = tv.tv_usec * 1000L + ((long)(t_ms) % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    pthread_mutex_lock(&c->lock);
    int r = 0;
    while (c->done == 0 && r == 0)
        r = pthread_cond_timedwait(&c->cond, &c->lock, &ts);
    int ret = (c->done > 0) ? 1 : 0;
    if (ret) c->done--;
    pthread_mutex_unlock(&c->lock);
    return (unsigned long)ret;
}

/* =========================================================
 * 9b. Semaphore (pthread condvar — avoids deprecated sem_init on macOS)
 * ========================================================= */
struct semaphore {
    pthread_mutex_t lock;
    pthread_cond_t  cond;
    int             count;
};
static inline void sema_init(struct semaphore *sem, int count) {
    pthread_mutex_init(&sem->lock, NULL);
    pthread_cond_init(&sem->cond, NULL);
    sem->count = count;
}
static inline void up(struct semaphore *sem) {
    pthread_mutex_lock(&sem->lock);
    sem->count++;
    pthread_cond_signal(&sem->cond);
    pthread_mutex_unlock(&sem->lock);
}
static inline void down(struct semaphore *sem) {
    pthread_mutex_lock(&sem->lock);
    while (sem->count <= 0)
        pthread_cond_wait(&sem->cond, &sem->lock);
    sem->count--;
    pthread_mutex_unlock(&sem->lock);
}
static inline int down_interruptible(struct semaphore *sem) {
    down(sem); return 0;
}

/* =========================================================
 * 10. Time, jiffies, msleep
 * ========================================================= */
#define HZ 1000

static inline unsigned long _get_jiffies(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (unsigned long)(tv.tv_sec * 1000UL + tv.tv_usec / 1000UL);
}
#define jiffies _get_jiffies()

static inline unsigned long msecs_to_jiffies(unsigned int ms) { return (unsigned long)ms; }
static inline unsigned int  jiffies_to_msecs(unsigned long j)  { return (unsigned int)j; }
#define time_after(a, b) ((long)(b) - (long)(a) < 0)

#define usleep_range(min, max) usleep((min))
#define msleep(ms)             usleep((unsigned int)(ms) * 1000u)
#define mdelay(ms)             usleep((unsigned int)(ms) * 1000u)
static inline unsigned long msleep_interruptible(unsigned int ms) {
    usleep(ms * 1000u);
    return 0;
}

/* =========================================================
 * 11. Timer (pthread-based)
 * ========================================================= */
#define COMPAT_TIMER_ARG_TYPE struct timer_list *

struct timer_list {
    void            (*function)(COMPAT_TIMER_ARG_TYPE);
    unsigned long     expires;
    volatile int      active;
    volatile int      cancelled;
    pthread_t         thread;
    int               thread_started;
    pthread_mutex_t   lock;
    pthread_cond_t    cond;
};

static void *_timer_thread_func(void *arg) {
    struct timer_list *t = (struct timer_list *)arg;
    pthread_mutex_lock(&t->lock);
    while (t->active && !t->cancelled) {
        unsigned long now = _get_jiffies();
        if ((long)(t->expires - now) <= 0) {
            t->active = 0;
            pthread_mutex_unlock(&t->lock);
            if (t->function) t->function(t);
            return NULL;
        }
        long ms_left = (long)(t->expires - now);
        struct timespec ts;
        struct timeval tv;
        gettimeofday(&tv, NULL);
        ts.tv_sec  = tv.tv_sec + ms_left / 1000;
        ts.tv_nsec = tv.tv_usec * 1000L + (ms_left % 1000) * 1000000L;
        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_cond_timedwait(&t->cond, &t->lock, &ts);
    }
    t->active = 0;
    pthread_mutex_unlock(&t->lock);
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
    pthread_mutex_init(&timer->lock, NULL);
    pthread_cond_init(&timer->cond, NULL);
}

static inline int mod_timer(struct timer_list *t, unsigned long expires) {
    /* Passive implementation: record the expiry and mark active, but do not
     * spawn a thread.  For the bulk_timer in ni_usb_gpib, the timer would
     * only fire if a transfer hangs longer than the libusb timeout (10 s),
     * which is already handled by libusb itself now that bulk URBs run
     * synchronously.  del_timer_sync() sees thread_started=0 and returns
     * without joining, making the timer pair essentially free. */
    pthread_mutex_lock(&t->lock);
    int was_active = t->active;
    t->expires   = expires;
    t->cancelled = 0;
    t->active    = 1;
    pthread_mutex_unlock(&t->lock);
    return was_active;
}

static inline int del_timer_sync(struct timer_list *t) {
    pthread_mutex_lock(&t->lock);
    int was_active = t->active;
    t->active    = 0;
    t->cancelled = 1;
    pthread_cond_signal(&t->cond);
    pthread_mutex_unlock(&t->lock);
    if (t->thread_started) {
        pthread_join(t->thread, NULL);
        t->thread_started = 0;
    }
    return was_active;
}
#define COMPAT_DEL_TIMER_SYNC(t) del_timer_sync(t)

#define COMPAT_FROM_TIMER(var, callback_timer, timer_fieldname) \
    container_of((struct timer_list *)(callback_timer), typeof(*(var)), timer_fieldname)

/* =========================================================
 * 12. USB structures and helpers
 * ========================================================= */

/* Internal pipe-type encoding */
#define _USB_PIPE_BULK_OUT  0x0000u
#define _USB_PIPE_BULK_IN   0x0100u
#define _USB_PIPE_INT_IN    0x0200u

#define usb_sndbulkpipe(dev, ep) (_USB_PIPE_BULK_OUT | ((unsigned int)(ep) & 0x7fu))
#define usb_rcvbulkpipe(dev, ep) (_USB_PIPE_BULK_IN  | ((unsigned int)(ep) & 0x7fu))
#define usb_rcvintpipe(dev, ep)  (_USB_PIPE_INT_IN   | ((unsigned int)(ep) & 0x7fu))
#define usb_rcvctrlpipe(dev, ep) (0x8000u)
#define usb_sndctrlpipe(dev, ep) (0x8001u)

/* USB control request bits */
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
    struct {
        uint16_t idVendor;
        uint16_t idProduct;
    } descriptor;
    struct usb_bus _bus_storage;
};

struct usb_interface {
    struct device     dev;
    struct usb_device *udev;
};

static inline struct usb_device *interface_to_usbdev(struct usb_interface *intf) {
    return intf->udev;
}
static inline void  usb_set_intfdata(struct usb_interface *i, void *d) { i->dev.driver_data = d; }
static inline void *usb_get_intfdata(struct usb_interface *i)           { return i->dev.driver_data; }
static inline void  usb_get_dev(struct usb_device *d) { (void)d; }
static inline void  usb_put_dev(struct usb_device *d) { (void)d; }
static inline int   usb_reset_configuration(struct usb_device *d) { (void)d; return 0; }
static inline void  usb_make_path(struct usb_device *d, char *buf, size_t len) {
    snprintf(buf, len, "/dev/usb/%d/%d", d->bus ? d->bus->busnum : 0, d->devnum);
}

static inline uint16_t USBID_TO_CPU(uint16_t id) { return id; }

/* Device ID table */
struct usb_device_id {
    uint16_t idVendor;
    uint16_t idProduct;
    int      bInterfaceNumber; /* -1 = don't care */
};
/* Note: no outer braces — callers write { USB_DEVICE(v,p) } themselves */
#define USB_DEVICE(vendor, product) \
    .idVendor = (vendor), .idProduct = (product), .bInterfaceNumber = -1
#define USB_DEVICE_INTERFACE_NUMBER(vendor, product, intf) \
    .idVendor = (vendor), .idProduct = (product), .bInterfaceNumber = (intf)

/* URB */
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
    /* internal */
    volatile int      cancelled;
    pthread_t         thread;
    int               thread_started;
    pthread_mutex_t   lock;
};

static inline struct urb *usb_alloc_urb(int iso_packets, int flags) {
    (void)iso_packets; (void)flags;
    struct urb *u = (struct urb *)calloc(1, sizeof(struct urb));
    if (u) pthread_mutex_init(&u->lock, NULL);
    return u;
}
static inline void usb_free_urb(struct urb *u) {
    if (u) { pthread_mutex_destroy(&u->lock); free(u); }
}

static inline void usb_fill_bulk_urb(struct urb *urb, struct usb_device *dev,
    unsigned int pipe, void *buf, int len,
    void (*complete)(struct urb *), void *ctx) {
    urb->dev = dev; urb->pipe = pipe;
    urb->transfer_buffer = buf; urb->transfer_buffer_length = len;
    urb->complete = complete;   urb->context = ctx;
}
static inline void usb_fill_int_urb(struct urb *urb, struct usb_device *dev,
    unsigned int pipe, void *buf, int len,
    void (*complete)(struct urb *), void *ctx, int interval) {
    urb->dev = dev; urb->pipe = pipe;
    urb->transfer_buffer = buf; urb->transfer_buffer_length = len;
    urb->complete = complete;   urb->context = ctx; urb->interval = interval;
}

static void *_urb_thread_func(void *arg) {
    struct urb *urb = (struct urb *)arg;
    unsigned int pipe_type = urb->pipe & 0xff00u;
    uint8_t ep = (uint8_t)(urb->pipe & 0x7fu);

    if (pipe_type == _USB_PIPE_INT_IN) {
        /* Interrupt endpoint: loop until cancelled */
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
        /* Bulk IN or OUT */
        if (pipe_type == _USB_PIPE_BULK_IN)
            ep |= LIBUSB_ENDPOINT_IN;
        int actual = 0;
        int r = libusb_bulk_transfer(urb->dev->handle, ep,
            (unsigned char *)urb->transfer_buffer,
            urb->transfer_buffer_length, &actual, 10000);
        if (urb->cancelled) {
            urb->status = -ECONNRESET;
        } else {
            urb->actual_length = actual;
            urb->status = (r == 0) ? 0 : -EIO;
        }
        if (urb->complete && !urb->cancelled)
            urb->complete(urb);
    }
    return NULL;
}

static inline int usb_submit_urb(struct urb *urb, int flags) {
    (void)flags;
    if (!urb || !urb->dev || !urb->dev->handle) return -ENODEV;
    urb->cancelled     = 0;
    urb->status        = 0;
    urb->actual_length = 0;

    unsigned int pipe_type = urb->pipe & 0xff00u;
    if (pipe_type == _USB_PIPE_INT_IN) {
        /* Skip interrupt endpoint polling.  In the kernel driver this endpoint
         * delivers async status notifications that wake threads waiting on
         * board->wait via wake_up_interruptible().  In this userspace port
         * nothing calls wait_event_interruptible_timeout(board->wait, ...), so
         * those wakeups are no-ops.  Running a background thread that hammers
         * libusb_interrupt_transfer() every 10 ms on the same device handle as
         * the bulk transfer threads causes USB-level serialisation on macOS/IOKit,
         * producing multi-second stalls.  Omitting the thread is safe and matches
         * the effective behaviour of NI488.2 / kernel nigpib. */
        return 0;
    } else {
        /* Bulk IN/OUT: run synchronously to avoid pthread_create overhead.
         * The driver always calls wait_for_completion() immediately after
         * usb_submit_urb(), so firing the completion callback here is safe —
         * wait_for_completion() will find done=1 and return without blocking. */
        uint8_t ep = (uint8_t)(urb->pipe & 0x7fu);
        if (pipe_type == _USB_PIPE_BULK_IN)
            ep |= LIBUSB_ENDPOINT_IN;
        int actual = 0;
        int r = libusb_bulk_transfer(urb->dev->handle, ep,
            (unsigned char *)urb->transfer_buffer,
            urb->transfer_buffer_length, &actual, 10000);
        urb->actual_length = actual;
        urb->status        = (r == 0) ? 0 : -EIO;
        if (urb->complete && !urb->cancelled)
            urb->complete(urb);
        /* thread_started remains 0 — usb_kill_urb() will be a no-op */
    }
    return 0;
}

static inline void usb_kill_urb(struct urb *urb) {
    if (!urb) return;
    urb->cancelled = 1;
    if (urb->thread_started) {
        pthread_join(urb->thread, NULL);
        urb->thread_started = 0;
    }
}

/* USB control message */
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
/* g_ni_usb_driver / g_ni_gpib_interface — filled by the module init */
extern struct usb_driver     *g_ni_usb_driver;
extern struct gpib_interface *g_ni_gpib_interface;

static inline int  usb_register(struct usb_driver *d) {
    g_ni_usb_driver = d; return 0;
}
static inline void usb_deregister(struct usb_driver *d) { (void)d; }

/* =========================================================
 * 13. task_struct stub
 * ========================================================= */
struct task_struct { int pid; };

/* =========================================================
 * 14. GPIB stubs (real types come from gpib_types.h later)
 * ========================================================= */
/* gpib_register_driver, gpib_unregister_driver, push_gpib_event, gpib_match_device_path
 * are declared in gpibP.h with proper types; implementations are in gpib_stubs.c */

/* =========================================================
 * 15. Utility Macros
 * ========================================================= */
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - __builtin_offsetof(type, member)))

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#ifndef min
#define min(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef max
#define max(x, y) ((x) > (y) ? (x) : (y))
#endif

/* Hex dump */
#define DUMP_PREFIX_NONE 0
static inline void print_hex_dump(const char *level, const char *prefix_str,
    int prefix_type, int rowsize, int groupsize, const void *buf, size_t len, bool ascii) {
    (void)level; (void)prefix_str; (void)prefix_type; (void)groupsize; (void)ascii;
    const uint8_t *p = (const uint8_t *)buf;
    for (size_t i = 0; i < len; i++) {
        printf("%02x ", p[i]);
        if ((i + 1) % (size_t)rowsize == 0) printf("\n");
    }
    if (len % (size_t)rowsize) printf("\n");
}

/* PCI stubs (referenced by gpibP.h) */
struct pci_dev { int unused; };

/* =========================================================
 * 16. Additional errno codes that macOS may not define
 * ========================================================= */
#ifndef ERESTARTSYS
#define ERESTARTSYS 512
#endif
#ifndef ENOTCONN
#define ENOTCONN    57
#endif
#ifndef ECOMM
#define ECOMM       70
#endif

#endif /* _MACOS_COMPAT_H_ */
