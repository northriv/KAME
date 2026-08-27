/* §13.145  Minimal shim so real GCC can compile allocator.cpp on macOS for
 * CODEGEN INSPECTION ONLY.  The macOS 26 SDK's <malloc/malloc.h> pulls in
 * <mach/message.h>, which uses the clang-only `xnu_static_assert_struct_size`
 * and makes GCC reject the SDK header outright -- that, not any project rule,
 * is what "kamepoolalloc refuses to build under GCC on macOS" (cdb70d2cf)
 * comes down to.  Objects built with this shim are for `objdump`, never to run.
 */
#ifndef KAME_GCC_MALLOC_SHIM_H_
#define KAME_GCC_MALLOC_SHIM_H_
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct _malloc_zone_t malloc_zone_t;
extern malloc_zone_t *malloc_zone_from_ptr(const void *ptr);
extern malloc_zone_t *malloc_default_zone(void);
extern void  malloc_zone_free(malloc_zone_t *zone, void *ptr);
extern void *malloc_zone_malloc(malloc_zone_t *zone, size_t size);
extern size_t malloc_size(const void *ptr);
extern size_t malloc_good_size(size_t size);
#ifdef __cplusplus
}
#endif
#endif

/* The one member the code touches is `size` (kame_malloc_size ->
 * z->size(z, p)).  A layout-faithful definition is not needed for codegen
 * inspection, only a compilable one with that member at a plausible offset;
 * this is why the resulting object must never be RUN.  */
#ifndef KAME_GCC_MALLOC_SHIM_ZONE_
#define KAME_GCC_MALLOC_SHIM_ZONE_
#ifdef __cplusplus
extern "C" {
#endif
struct _malloc_zone_t {
    void *reserved1, *reserved2;
    size_t (*size)(struct _malloc_zone_t *zone, const void *ptr);
    void  *(*malloc)(struct _malloc_zone_t *zone, size_t size);
    void  *(*calloc)(struct _malloc_zone_t *zone, size_t n, size_t sz);
    void  *(*valloc)(struct _malloc_zone_t *zone, size_t size);
    void   (*free)(struct _malloc_zone_t *zone, void *ptr);
    void  *(*realloc)(struct _malloc_zone_t *zone, void *ptr, size_t size);
    void   (*destroy)(struct _malloc_zone_t *zone);
    const char *zone_name;
};
extern void *malloc_zone_realloc(malloc_zone_t *zone, void *ptr, size_t size);
extern void *malloc_zone_calloc(malloc_zone_t *zone, size_t n, size_t sz);
#ifdef __cplusplus
}
#endif
#endif
