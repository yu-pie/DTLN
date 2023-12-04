#include "alg_mem_pool.h"

struct alg_mem_pool *alg_mem_pool_init(int blk_cnt, int blk_size)
{
    int total_size;
    void *base_mem;
    struct alg_mem_pool *p_mem_pool;
    void *used;

    ALG_ASSERT(blk_cnt && blk_size);

    total_size = blk_cnt * blk_size;
    base_mem = ALG_MALLOC(total_size);
    if (base_mem == NULL) {
        ALG_PRINT("%s malloc fail!\n", __func__);
        return NULL;
    }

    used = ALG_MALLOC(blk_cnt);
    if (used == NULL) {
        ALG_PRINT("%s malloc fail!\n", __func__);
        ALG_FREE(base_mem);
        return NULL;
    }

    p_mem_pool = (struct alg_mem_pool *)ALG_MALLOC(sizeof(struct alg_mem_pool));
    if (p_mem_pool == NULL) {
        ALG_PRINT("%s malloc fail!\n", __func__);
        ALG_FREE(base_mem);
        ALG_FREE(used);
        return NULL;
    }

    ALG_MEMSET(used, 0, blk_cnt);
    ALG_MEMSET(p_mem_pool, 0, sizeof(struct alg_mem_pool));
    p_mem_pool->blk_cnt = blk_cnt;
    p_mem_pool->blk_size = blk_size;
    p_mem_pool->total_size = blk_cnt * blk_size;
    p_mem_pool->base_mem = base_mem;
    p_mem_pool->used = used;
    ALG_LOCK_INIT(p_mem_pool->lock);

    return p_mem_pool;
}

void alg_mem_pool_deinit(struct alg_mem_pool *p_mem_pool)
{
    ALG_ASSERT(p_mem_pool != NULL);
    ALG_ASSERT(p_mem_pool->base_mem != NULL);
    ALG_ASSERT(p_mem_pool->used != NULL);

    ALG_FREE(p_mem_pool->base_mem);
    ALG_FREE(p_mem_pool->used);
    ALG_LOCK_FINI(p_mem_pool->lock);
    ALG_FREE(p_mem_pool);
}

void *alg_request_buf(struct alg_mem_pool *p_mem_pool)
{
    int idx;

    ALG_ASSERT(p_mem_pool != NULL);
    ALG_LOCK(p_mem_pool->lock);
    for (idx = 0; idx < p_mem_pool->blk_cnt; idx++) {
        if (p_mem_pool->used[idx])
            continue;

        p_mem_pool->used[idx] = 1;
        ALG_UNLOCK(p_mem_pool->lock);
        return (p_mem_pool->base_mem + idx * p_mem_pool->blk_size);
    }
    ALG_UNLOCK(p_mem_pool->lock);

    ALG_PRINT("%s fail!\n", __func__);
    return NULL;
}

int alg_release_buf(struct alg_mem_pool *p_mem_pool, void *buf)
{
    int idx, ret;

    ALG_ASSERT(p_mem_pool != NULL);
    ALG_ASSERT(p_mem_pool->base_mem <= buf && 
            buf < (p_mem_pool->base_mem + p_mem_pool->total_size));

    idx = (buf - p_mem_pool->base_mem) / p_mem_pool->blk_size;
    ALG_LOCK(p_mem_pool->lock);
    if (p_mem_pool->used[idx]) {
        p_mem_pool->used[idx] = 0;
        ret = 0;
    } else {
        ALG_PRINT("%s fail! idx:%d\n", __func__, idx);
        ret = -1;
    }
    ALG_UNLOCK(p_mem_pool->lock);

    return ret;
}
