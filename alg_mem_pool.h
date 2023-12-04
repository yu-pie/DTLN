#ifndef __ALG_MEM_POOL_H__
#define __ALG_MEM_POOL_H__

#include "alg_module.h"

struct alg_mem_pool {
    int blk_cnt;
    int blk_size;
    int total_size;
    void *base_mem;
    char *used;
    ALG_LOCK_TYPE lock;
};

struct alg_mem_pool *alg_mem_pool_init(int blk_cnt, int blk_size);
void alg_mem_pool_deinit(struct alg_mem_pool *p_mem_pool);
void *alg_request_buf(struct alg_mem_pool *p_mem_pool);
int alg_release_buf(struct alg_mem_pool *p_mem_pool, void *buf);

#endif
