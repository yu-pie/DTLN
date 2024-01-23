#include "npu.h"
#include "element_wise.h"
#include "npu_perf.h"
#include <stdio.h>

#pragma GCC push_options
#pragma GCC optimize("O2")

#define LOG_TAG "GRU"

// #define GRU_DEBUG
#define PERF_TRACE 0

// #define FP_CORE 0
// #define DMA_CORE 1
// #define MATMUL_CORE 2
// #define SCHEDULE_CORE 3
#define FP_CORE       0
#define SCHEDULE_CORE 0
#define DMA_CORE      0
#define MATMUL_CORE   0

#define SINGLE_CORE_MODE (FP_CORE == SCHEDULE_CORE && SCHEDULE_CORE == DMA_CORE && DMA_CORE == MATMUL_CORE)

#define MAX_ARGS_NUM 8

typedef volatile enum {
    TASK_CREATE = 0,
    TASK_WAITING,
    TASK_RUNNING,
    TASK_FINISH,
} Task_status;

typedef volatile struct {
#ifdef GRU_DEBUG
    /// @brief task name
    char *task_name;
#endif

    /// @brief task function pointer
    void (*task_func)(u32 *);
    /// @brief task function arguments
    u32 args[MAX_ARGS_NUM];
    /// @brief task status pointer. @sa Task_status
    Task_status status;
    /// @brief pre task status pointer, this is used for task dependency, if pre_status is not TASK_FINISH, this task will not be executed.
    Task_status *pre_task_status;
} Task_t;

// wfi is more power efficient than nop
#define NOP_DELAY(x) \
    do { \
        for (volatile uint32_t i = 0; i < x; i++) { \
            __asm__ volatile("wfi"); \
        } \
    } while (0)

// #define GRU_WAIT() NOP_DELAY(0)
// current test no delay is more efficient
#define GRU_WAIT()

#define WAIT_TASK_FINISH(p_task_status) \
    do { \
        while (*(p_task_status) != TASK_FINISH) { \
            GRU_WAIT(); \
        } \
    } while (0)

#define GRU_TASK_QUEUE_DEPTH 8
/// @brief task queue
__attribute__((section(".ccdata"))) __aligned(4) static volatile Task_t task_queue[NPU_CORE_CNT][GRU_TASK_QUEUE_DEPTH];
/// @brief task queue head
__attribute__((section(".ccdata"))) __aligned(4) static volatile uint8_t task_queue_head[NPU_CORE_CNT];
/// @brief task queue tail
__attribute__((section(".ccdata"))) __aligned(4) static volatile uint8_t task_queue_tail[NPU_CORE_CNT];
/// @brief task queue exit flag
__attribute__((section(".ccdata"))) __aligned(4) static volatile uint8_t task_queue_exit = 0;
/// @brief task queue start flag
__attribute__((section(".ccdata"))) __aligned(4) static volatile uint8_t task_queue_start = 0;
__attribute__((section(".ccdata"))) __aligned(4) static volatile i32 core_running[NPU_CORE_CNT];

// #define NPU_LOG_DEBUG
// #define NPU_LOG_INFO
// #define NPU_LOG_WARNING
// #define NPU_LOG_ERROR

#ifdef GRU_DEBUG
__attribute__((section(".ccdata"))) __aligned(4) static char log_buf[256];
#define GRU_NPU_PRINTF(fmt, ...) \
    do { \
        if (GetCoreId() == 0) { \
            snprintf(log_buf, 255, "" fmt, ##__VA_ARGS__); \
            NPU_PRINTF("%s", log_buf); \
        } \
    } while (0)
#define GRU_NPU_LOG(fmt, ...) GRU_NPU_PRINTF("[%s] " fmt, LOG_TAG, ##__VA_ARGS__)
#define GRU_NPU_ASSERT(x) \
    do { \
        if ((x) == 0) { \
            NPU_ASSERT(x); \
        } \
    } while (0)
#else
#define GRU_NPU_PRINTF(fmt, ...) NPU_PRINTF(fmt, ##__VA_ARGS__)
#define GRU_NPU_LOG(fmt, ...)
#define GRU_NPU_ASSERT(x) NPU_ASSERT(x)
#endif

__attribute__((section(".ccdata"))) __aligned(4) static Task_status _TASK_FINISH = TASK_FINISH;

/**
 * @brief initialize gru multi core task
 *
 */
static void gru_multi_core_init(void)
{
    task_queue_exit = 0;
    for (int i = 0; i < NPU_CORE_CNT; i++) {
        for (int j = 0; j < GRU_TASK_QUEUE_DEPTH; j++) {
            task_queue[i][j].status = TASK_FINISH;
        }
    }
    GRU_NPU_LOG("gru_multi_core_init finish\n");
}

static void gru_process(u32 *param);

static void print_core_state(void)
{
    for (int i = 1; i < NPU_CORE_CNT; i++) {
        GRU_NPU_LOG("core %d running count: %d\n", i, core_running[i]);
        for (int j = 0; j < GRU_TASK_QUEUE_DEPTH; j++) {
            GRU_NPU_LOG("core %d task %d status: %d\n", i, j, task_queue[i][j].status);
        }
    }
}

/**
 * @brief start gru multi core task
 * If core_id is SCHEDULE_CORE, it will run gru_process function. Otherwise, it will loop to run task in task queue, until task_queue_exit is set.
 * Task execution is ordered by task_queue_head and task_queue_tail, task_queue_head is the index of task to be executed, task_queue_tail is the index of task to be submitted.
 * If current task requires pre task to be finished, it will wait until pre task is finished.
 *
 * @param param
 */
static void gru_multi_core(void *param)
{
    i32 core_id = GetCoreId();

    if (core_id != SCHEDULE_CORE && core_id != FP_CORE && core_id != DMA_CORE && core_id != MATMUL_CORE) {
        return;
    }

    // GRU_NPU_LOG("NPU_CORE_CNT=%d, GRU_TASK_QUEUE_DEPTH=%d\n", NPU_CORE_CNT, GRU_TASK_QUEUE_DEPTH);
    GRU_NPU_LOG("gru_multi_core start on core %d, core_running addr: %p\n", core_id, core_running);
    core_running[core_id] = 1;

    while (!task_queue_exit) {
        if (task_queue_head[core_id] == task_queue_tail[core_id] && task_queue[core_id][task_queue_tail[core_id]].status == TASK_FINISH) {
            GRU_WAIT();
            continue;
        }
        Task_t *task = task_queue[core_id] + task_queue_head[core_id];
        if (task->pre_task_status != NULL && *(task->pre_task_status) != TASK_FINISH) {
            GRU_WAIT();
            continue;
        }

        task->status = TASK_RUNNING;
        // GRU_NPU_LOG("task %s start on core %d\n", task->task_name, core_id);
        task->task_func((u32 *)task->args);
        // GRU_NPU_LOG("task %s finish on core %d\n", task->task_name, core_id);
        task->status = TASK_FINISH;

        task_queue_head[core_id] = (task_queue_head[core_id] + 1) % GRU_TASK_QUEUE_DEPTH;
    }
    core_running[core_id] = 0;
    GRU_NPU_LOG("gru_multi_core exit on core %d\n", core_id);
}

/**
 * @brief call function with variable arguments
 *
 * @param task_func function pointer
 * @param args_num arguments number
 * @param ... arguments
 */
static void va_args_func_call(void (*task_func)(u32 *), u32 args_num, ...)
{
    va_list args;
    va_start(args, args_num);
    u32 args_array[MAX_ARGS_NUM];
    for (int i = 0; i < args_num; i++) {
        args_array[i] = va_arg(args, u32);
    }
    va_end(args);
#if PERF_TRACE
    u32 start_cycle = (u32)NpuPerGetLo();
#endif
    task_func(args_array);
#if PERF_TRACE
    u32 end_cycle  = (u32)NpuPerGetLo();
    u32 used_cycle = end_cycle - start_cycle;
    GRU_NPU_PRINTF("spend time: %u ns\n", used_cycle * 1000 / NPU_PFF_FREQ_MHZ);
#endif
}

/**
 * @brief submit task to task queue
 *
 * @param task_name task name
 * @param task_func task function pointer
 * @param args task function arguments
 * @param status task status pointer
 * @param core_id core id
 * @param block 1: block until task is finished, 0: not block
 * @param pre_task_status pre task status pointer
 * @param args_num arguments number
 */
#ifdef GRU_DEBUG
static Task_status *submit_task(char *task_name, void (*task_func)(u32 *), i32 core_id, i32 block, Task_status *pre_task_status, u32 args_num, ...)
{
    GRU_NPU_LOG("submit_task, task_name=%s, core_id=%d, block=%d, args_num=%lu\n", task_name, core_id, block, args_num);
    GRU_NPU_ASSERT(task_func != NULL);
    GRU_NPU_ASSERT(core_id >= 0 && core_id < NPU_CORE_CNT);
#else
static Task_status *submit_task(void (*task_func)(u32 *), i32 core_id, i32 block, Task_status *pre_task_status, u32 args_num, ...)
{
#endif
    GRU_NPU_LOG("current core id: %d, core_running[%d]: %d, task_queue_exit: %d\n", GetCoreId(), core_id, core_running[core_id], task_queue_exit);
    // if core_id is self and run immediately is possible, run task immediately.
    if (core_id == GetCoreId() && core_running[core_id]) {
        GRU_NPU_LOG("run task %s immediately\n", task_name);
        if (pre_task_status) WAIT_TASK_FINISH(pre_task_status);

        va_list args;
        va_start(args, args_num);
        volatile u32 args_array[MAX_ARGS_NUM];
#ifdef GRU_DEBUG
        GRU_NPU_LOG("args: ");
#endif
        for (int i = 0; i < args_num; i++) {
            args_array[i] = va_arg(args, u32);
#ifdef GRU_DEBUG
            GRU_NPU_PRINTF("0x%lx, ", args_array[i]);
#endif
        }
#ifdef GRU_DEBUG
        GRU_NPU_PRINTF("\n");
#endif
        task_func(args_array);
        // va_args_func_call(task_func, args_num, args);
        va_end(args);

        GRU_NPU_LOG("run task %s immediately finish\n", task_name);
        return &_TASK_FINISH;
    }

    while (task_queue_head[core_id] == ((task_queue_tail[core_id] + 1) % GRU_TASK_QUEUE_DEPTH)) {
        // GRU_NPU_LOG("task queue is full on core %d\n", core_id);
        GRU_WAIT();
    }
    Task_t *task = task_queue[core_id] + task_queue_tail[core_id];
#ifdef GRU_DEBUG
    task->task_name = task_name;
#endif
    task->task_func = task_func;

    va_list args;
    va_start(args, args_num);
#ifdef GRU_DEBUG
    GRU_NPU_LOG("args: ");
#endif
    for (int i = 0; i < args_num; i++) {
        task->args[i] = va_arg(args, u32);
#ifdef GRU_DEBUG
        GRU_NPU_PRINTF("0x%lx, ", task->args[i]);
#endif
    }
#ifdef GRU_DEBUG
    GRU_NPU_PRINTF("\n");
#endif
    va_end(args);

    // avoid task rely on itself
    GRU_NPU_ASSERT(pre_task_status != &task->status);

    task->status          = TASK_WAITING;
    task->pre_task_status = pre_task_status;
    // GRU_NPU_LOG("test, *status: %d\n", *status);
    task_queue_tail[core_id] = (task_queue_tail[core_id] + 1) % GRU_TASK_QUEUE_DEPTH;
    if (block) {
        while (task->status != TASK_FINISH) {
            // print_core_state();
            GRU_WAIT();
        }
        GRU_NPU_LOG("task run finish in block mode\n");
    } else {
        // GRU_NPU_LOG("task submited\n");
    }
    return (Task_status *)&task->status;
}

#ifdef GRU_DEBUG
#define SUBMIT_TASK(func, core_id, block, pre_task_status, args_num, ...) \
    ({ \
        Task_status *p; \
        if (SINGLE_CORE_MODE) { \
            if (PERF_TRACE) \
                GRU_NPU_PRINTF("task %s ", #func); \
            va_args_func_call(func, args_num, ##__VA_ARGS__); \
            p = &_TASK_FINISH; \
        } else { \
            p = submit_task(#func, func, core_id, block, pre_task_status, args_num, ##__VA_ARGS__); \
        } \
        p; \
    })
#else
#define SUBMIT_TASK(func, core_id, block, pre_task_status, args_num, ...) \
    ({ \
        Task_status *p; \
        if (SINGLE_CORE_MODE) { \
            if (PERF_TRACE) \
                NPU_PRINTF("task %s ", #func); \
            va_args_func_call(func, args_num, ##__VA_ARGS__); \
            p = &_TASK_FINISH; \
        } else { \
            p = submit_task(func, core_id, block, pre_task_status, args_num, ##__VA_ARGS__); \
        } \
        p; \
    })
#endif

/**
 * @brief dma task function
 *
 * @param param param[0](u32): l1 addr, param[1](u32): l2 addr, param[2](u32): len, param[3](int): type
 */
static void dma_1d(u32 *param)
{
    u32 dst_addr = ((u32 *)param)[0];
    u32 src_addr = ((u32 *)param)[1];
    u32 len      = ((u32 *)param)[2];
    int type     = ((int *)param)[3];

    // GRU_NPU_LOG("dma_1d, dst_addr=0x%x, src_addr=0x%x, len=%d, type=%d\n", dst_addr, src_addr, len, type);

    Dma1d(dst_addr, src_addr, len, type);
}

static inline float _get_quant_input_dynamic_scale(u32 len, float *in)
{
    register float tmpmax = *((float *)in);
    register float tmpmin = *((float *)in);
    // GRU_NPU_LOG("in 0, 1, 2=%f, %f, %f\n", in[0], in[1], in[2]);
    for (int i = 1; i < len; i++) {
        tmpmax = NPU_MAX(tmpmax, *((float *)in + i));
        tmpmin = NPU_MIN(tmpmin, *((float *)in + i));
    }
    if (tmpmax < 0.0f)
        tmpmax = -tmpmax;
    if (tmpmin < 0.0f)
        tmpmin = -tmpmin;
    register float val = NPU_MAX(tmpmax, tmpmin);
    return (float)(val / 127.0f);
}

/**
 * @brief convert int32 to fp32
 *
 * @param param param[0](void*): input/output address, param[1](u32): length, param[2](float): scale1, param[3](float): scale2
 */
static void int32_to_fp32(u32 *param)
{
    register i32  *p     = ((i32 **)param)[0];
    register u32   len   = ((u32 *)param)[1];
    register float scale = ((float *)param)[2] * ((float *)param)[3];

    // GRU_NPU_LOG("scale1: %f, scale2: %f\n", ((float *)param)[2], ((float *)param)[3]);
    // GRU_NPU_LOG("scale: %f\n", scale);
    // DeQuant(p, NPU_INT32, p, NPU_FP32, len, scale, 0);
    for (int i = 0; i < len; i++) {
        *((float *)p) = *p * scale;
        p++;
    }
}

/**
 * @brief convert fp32 to int8
 *
 * @param param param[0](float*): input, param[1](i8*): output address, param[2](u32): length, param[3](float*): scale pointer(read in static scale, write in dynamic scale), param[4](i32): 1-dynamic scale, 0-static scale
 */
static void fp32_to_int8(u32 *param)
{
    register float *input         = ((float **)param)[0];
    register i8    *output        = ((i8 **)param)[1];
    const u32    len           = ((u32 *)param)[2];
    float *p_scale       = ((float **)param)[3];
    const i32             dynamic_scale = ((i32 *)param)[4];

    register float scale = *p_scale;
    if (dynamic_scale) {
        scale = _get_quant_input_dynamic_scale(len, input);
    } 
    *p_scale = scale;
    // GRU_NPU_LOG("fp32_to_int8, input=%p, output=%p, len=%lu, dynamic_scale=%d, scale=%f\n",
    //             input, output, len, dynamic_scale, *p_scale);
    GRU_NPU_LOG("fp32_to_int8, input=%p, output=%p, len=%lu, dynamic_scale=%d, scale*10000=%d\n",
                input, output, len, dynamic_scale, (i32)(*p_scale * 10000));
    // Quant(input, NPU_FP32, output, NPU_INT8, len, scale, 0);
    scale = 1.0f / scale;
    for (int i = 0; i < len; i++) {
        float tempfp = (*input++) * scale;
        i32 temp32 = (i32)NPU_ROUND(tempfp);
        *output++ = (i8)NPU_CLAMP_INT8(temp32);
    }
}

/**
 * @brief element wise mul for fp32
 *
 * @param param param[0](float*): input/output address, param[1](float*): input address, param[2](u32): length
 */
static void fp_element_wise_mul(u32 *param)
{
    register float *p1  = ((float **)param)[0];
    register float *p2  = ((float **)param)[1];
    register u32    len = ((u32 *)param)[2];

    for (int i = 0; i < len; i++) {
        *p1++ *= *p2++;
    }
}

/**
 * @brief element wise add for fp32
 *
 * @param param param[0](float*): input/output address, param[1](float*): input address, param[2](u32): length
 */
static void fp_element_wise_add(u32 *param)
{
    register float *p1  = ((float **)param)[0];
    register float *p2  = ((float **)param)[1];
    u32    len = ((u32 *)param)[2];

    for (int i = 0; i < len; i++) {
        *p1++ += *p2++;
    }
}

/**
 * @brief gru matmul. input[M, K] * weight[K, N] + bias[M, N] -> output[M, N]
 *
 * @param param param[0](u32): input base address, param[1](u32): weight base address, param[2](u32): bias base address, param[3](u32): output base address, param[4](u32): M, param[5](u32): K, param[6](u32): N
 */
static void gru_matmul(u32 *param)
{
    u32 input_base  = ((u32 *)param)[0];
    u32 weight_base = ((u32 *)param)[1];
    u32 bias_base   = ((u32 *)param)[2];
    u32          output_base = ((u32 *)param)[3];
    u32 M           = ((u32 *)param)[4];
    u32 K           = ((u32 *)param)[5];
    u32 N           = ((u32 *)param)[6];

    // GRU_NPU_LOG("gru_matmul, input_base=0x%x, weight_base=0x%x, bias_base=0x%x, output_base=0x%x, M=%d, K=%d, N=%d\n",
    //         input_base, weight_base, bias_base, output_base, M, K, N);

    u32 cur_output_base = output_base;
    for (int i = 0; i < M; i++) {
        // GRU_NPU_LOG("gru_matmul, input_base=0x%x, weight_base=0x%x, bias_base=0x%x, cur_output_base=0x%x, M=%d, K=%d, N=%d\n",
        //             input_base, weight_base, bias_base, cur_output_base, M, K, N);
        TsmeFcJobUpdateBase(0, weight_base, bias_base, input_base, cur_output_base);
        TsmeStartJob(0);
        // bias add not support if output INT32, use software to add bias
        if (i != 0) {
            // add bias for previous iteration
            register i32 *p = (i32 *)cur_output_base - N;
            register i8  *b = (i8 *)bias_base;
            for (int j = 0; j < N; j++) {
                *p++ += *b++;
            }
        }

        input_base += K * NPU_TYPE_WIDTH(NPU_INT8);
        cur_output_base += N * NPU_TYPE_WIDTH(NPU_INT32);
        TsmeWaitJobDone(0);
    }
    // bias add not support if output INT32, use software to add bias
    // prepare bias for latest iteration
    register i32 *p = (i32 *)cur_output_base - N;
    register i8  *b = (i8 *)bias_base;
    for (int j = 0; j < N; j++) {
        *p++ += *b++;
    }

    // GRU_NPU_LOG("gru_matmul finish\n");
}

/**
 * @brief initialize tsme layer
 *
 * @param param not used
 */
static void init_tsme(u32 *param)
{
    // config tsme layer
    TsmeCtrl_t ctrl = TSME_CTRL_DEFAULT_CONFIG();
    ctrl.tsme_mode  = TSME_FC;
    TsmeLayerConfig(&ctrl);

    // pre config for matmul Xt * (W[zrh]^T) + Wb[zrh]
    TsmeFcJob_t tsme_job;
    tsme_job.fci      = 128;
    tsme_job.fco      = 128;
    tsme_job.count    = 1;
    tsme_job.act      = ACT_NONE;
    tsme_job.has_bias = 1;
    tsme_job.shift    = 0;
    tsme_job.nopost   = 1;
    TsmeFcJobConfig(&tsme_job, 0);
}

/**
 * @brief update tsme io size
 *
 * @param param param[1](u32): input size, param[2](u32): output size
 */
static void update_tsme_io_size(u32 *param)
{
    u32 input_size  = ((u32 *)param)[0];
    u32 output_size = ((u32 *)param)[1];

    TsmeUpdateInOutSize(0, input_size, output_size);
}

/**
 * @brief float sigmoid. replace input with sigmoid(input)
 *
 * @param param param[0](float*): input address, param[2](u32): length
 */
static void fp_sigmoid(u32 *param)
{
    float *p   = ((float **)param)[0];
    u32    len = ((u32 *)param)[1];

    Sigmoid(p, NPU_FP32, p, NPU_FP32, len);
}

/**
 * @brief float tanh. replace input with tanh(input)
 *
 * @param param param[0](float*): input address, param[2](u32): length
 */
static void fp_tanh(u32 *param)
{
    float *p   = ((float **)param)[0];
    u32    len = ((u32 *)param)[1];

    Tanh(p, NPU_FP32, p, NPU_FP32, len);
}

/**
 * @brief update hidden state. Ht = zt * Ht-1 * scale + (1 - zt) * ht
 *
 * @param param param[0](float*): Ht-1/Ht, param[1](float*): zt, param[2](float*): ht, param[3](u32): length
 */
static void gru_update_hidden_state(u32 *param)
{
    register float *Ht  = ((float **)param)[0];
    register float *zt  = ((float **)param)[1];
    register float *ht  = ((float **)param)[2];
    register u32    len = ((u32 *)param)[3];

    for (int i = 0; i < len; i++) {
        float Ht_i = *Ht;
        float zt_i = *zt++;
        float ht_i = *ht++;
        *Ht++               = Ht_i * zt_i + ht_i * (1 - zt_i);
    }
}

/**
 * @brief check if gru layer can be full cache in L1
 *
 * @param layer Grulayer_t instance
 * @return i32 1: can be full cache, 0: can not be full cache
 */
static i32 Gru_fullcache_check(GruLayer_t *layer)
{
    i32 batch_size  = layer->batch_size;
    i32 input_size  = layer->input_size;
    i32 hidden_size = layer->hidden_size;

    u32 l1_used = 0;

    u32 w_size  = NPU_ALIGN_64(hidden_size * input_size * NPU_TYPE_WIDTH(layer->w_type));
    u32 wb_size = NPU_ALIGN_64(hidden_size * NPU_TYPE_WIDTH(layer->b_type));
    u32 r_size  = NPU_ALIGN_64(hidden_size * hidden_size * NPU_TYPE_WIDTH(layer->w_type));
    u32 rb_size = NPU_ALIGN_64(hidden_size * NPU_TYPE_WIDTH(layer->b_type));
    // model weights size.
    u32 weights_size = (w_size + wb_size + r_size + rb_size) * 3;
    l1_used += weights_size;
    // hidden state size.
    u32 h_size = NPU_ALIGN_64(batch_size * hidden_size * NPU_TYPE_WIDTH(NPU_FP32));
    // Ht size.
    l1_used += h_size;
    // quant Ht size.
    u32 h_quant_size = NPU_ALIGN_64(batch_size * hidden_size * NPU_TYPE_WIDTH(NPU_INT8));
    l1_used += h_quant_size;
    // zt+rt+ht size.
    u32 zrh_size = h_size * 3;
    l1_used += zrh_size;
    if (layer->pre.quant == QUANT_QUANT) {
        // float input size.
        u32 input_fp_size = NPU_ALIGN_64(batch_size * input_size * NPU_TYPE_WIDTH(NPU_FP32));
        l1_used += input_fp_size;
    } else {
        // Xt size.
        u32 Xt_size = NPU_ALIGN_64(batch_size * input_size * NPU_TYPE_WIDTH(layer->i_type));
        l1_used += Xt_size;
    }

    return l1_used <= L1_CNN_Size;
}

/**
 * @brief check if gru layer can be run in normal mode
 *
 * @param layer Grulayer_t instance
 * @return i32 1: can be run in normal mode, 0: can not be run in normal mode
 */
static i32 Gru_normal_mem_check(GruLayer_t *layer)
{
    i32 batch_size  = layer->batch_size;
    i32 input_size  = layer->input_size;
    i32 hidden_size = layer->hidden_size;

    u32 l1_used = 0;

    u32 w_size  = NPU_ALIGN_64(hidden_size * input_size * NPU_TYPE_WIDTH(layer->w_type));
    u32 wb_size = NPU_ALIGN_64(hidden_size * NPU_TYPE_WIDTH(layer->b_type));
    u32 r_size  = NPU_ALIGN_64(hidden_size * hidden_size * NPU_TYPE_WIDTH(layer->w_type));
    u32 rb_size = NPU_ALIGN_64(hidden_size * NPU_TYPE_WIDTH(layer->b_type));
    // model weights size.
    u32 part_weights_size = w_size + wb_size + r_size + rb_size;
    l1_used += part_weights_size;
    // hidden state size.
    u32 h_size = NPU_ALIGN_64(batch_size * hidden_size * NPU_TYPE_WIDTH(NPU_FP32));
    // Ht size.
    l1_used += h_size;
    // quant Ht size.
    u32 h_quant_size = NPU_ALIGN_64(batch_size * hidden_size * NPU_TYPE_WIDTH(NPU_INT8));
    l1_used += h_quant_size;
    // required buffer size.
    l1_used += h_size * 2;
    if (layer->pre.quant == QUANT_QUANT) {
        // float input size.
        u32 input_fp_size = NPU_ALIGN_64(batch_size * input_size * NPU_TYPE_WIDTH(NPU_FP32));
        l1_used += input_fp_size;
    } else {
        // Xt size.
        u32 Xt_size = NPU_ALIGN_64(batch_size * input_size * NPU_TYPE_WIDTH(layer->i_type));
        l1_used += Xt_size;
    }

    return l1_used <= L1_CNN_Size;
}

/**
 * @brief Initialize GRU layer
 *
 * @param layer gru layer pointer. @sa GruLayer_t
 * @param model_weight model weight pointer
 * @return GruLayerHnd_t gru layer handle instance
 */
GruLayerHnd_t GruLayerInit(GruLayer_t *layer, void *model_weight)
{
    GRU_NPU_LOG("init, layer=%p, model_weight=%p\n",
                layer, model_weight);
    NPU_ASSERT(layer != NULL);
    NPU_ASSERT((layer->pre.quant == QUANT_NONE) || (layer->pre.quant == QUANT_QUANT));
    NPU_ASSERT((layer->post.quant == QUANT_NONE) || (layer->post.quant == QUANT_QUANT));
    NPU_ASSERT(layer->i_type == NPU_INT8);
    NPU_ASSERT(layer->direction & GRU_BIDIRECTIONAL);

    GruLayerHnd_t hnd = NPU_MALLOC(sizeof(GruLayerMgr_t));
    NPU_ASSERT(hnd != NULL);
    NPU_MEMSET(hnd, 0, sizeof(GruLayerMgr_t));

    /* initialize layer infomation and weight */
    hnd->layer = layer;
    // weights size for each gate.
    u32 w_size = layer->hidden_size * layer->input_size * NPU_TYPE_WIDTH(layer->w_type);
    // bias size for each gate.
    u32 b_size = layer->hidden_size * NPU_TYPE_WIDTH(layer->b_type);
    // weights size for update, reset, and hidden gates.
    u32 W_size = w_size * 3;
    // bias size for update, reset, and hidden gates.
    u32 B_size = b_size * 3;
    // hidden state size.
    u32 h_size = layer->batch_size * layer->hidden_size * NPU_TYPE_WIDTH(NPU_FP32);
    // input hidden state size.
    u32 init_h_size = layer->batch_size * layer->hidden_size * NPU_TYPE_WIDTH(layer->h_type);

    u32 model_offset = 0;
    // set model default offset if required.
    if (layer->direction & GRU_FORWARD) {
        if (!layer->W_offset)
            layer->W_offset = model_offset;
        model_offset += W_size;
        if (!layer->R_offset)
            layer->R_offset = model_offset;
        model_offset += W_size;
        if (!layer->Wb_offset)
            layer->Wb_offset = model_offset;
        model_offset += B_size;
        if (!layer->Rb_offset)
            layer->Rb_offset = model_offset;
        model_offset += B_size;
        if (!layer->init_H_offset) {
            layer->init_H_offset = model_offset;
            model_offset += layer->hidden_size * NPU_TYPE_WIDTH(layer->pre.qi_type);
        }
    }
    if (layer->direction & GRU_BACKWARD) {
        if (!layer->WB_offset)
            layer->WB_offset = model_offset;
        model_offset += W_size;
        if (!layer->RB_offset)
            layer->RB_offset = model_offset;
        model_offset += W_size;
        if (!layer->WBb_offset)
            layer->WBb_offset = model_offset;
        model_offset += B_size;
        if (!layer->RBb_offset)
            layer->RBb_offset = model_offset;
        model_offset += B_size;
        if (!layer->init_HB_offset) {
            layer->init_HB_offset = model_offset;
            model_offset += layer->hidden_size * NPU_TYPE_WIDTH(layer->pre.qi_type);
        }
    }

    // set address of weights and bias.
    hnd->W   = (void *)((i32)model_weight + layer->W_offset);
    hnd->R   = (void *)((i32)model_weight + layer->R_offset);
    hnd->Wb  = (void *)((i32)model_weight + layer->Wb_offset);
    hnd->Rb  = (void *)((i32)model_weight + layer->Rb_offset);
    hnd->WB  = (void *)((i32)model_weight + layer->WB_offset);
    hnd->RB  = (void *)((i32)model_weight + layer->RB_offset);
    hnd->WBb = (void *)((i32)model_weight + layer->WBb_offset);
    hnd->RBb = (void *)((i32)model_weight + layer->RBb_offset);

    if (layer->direction & GRU_FORWARD) {
        /* malloc for hidden state */
        hnd->H = NPU_MALLOC(h_size);
        NPU_ASSERT(hnd->H != NULL);
        if (layer->init_H_offset == -1) {
            // if init_H_offset is -1, set init hidden state to 0.
            NPU_MEMSET(hnd->H, 0, h_size);
        } else {
            void *init_H = (void *)((i32)model_weight + layer->init_H_offset);
            // convert init hidden state to float
            DeQuant(init_H, NPU_INT8, hnd->H, NPU_FP32, layer->batch_size * layer->hidden_size, layer->q_init_H.scale, 0);
        }
    }

    if (layer->direction & GRU_BACKWARD) {
        /* malloc for hidden state for backward direction */
        hnd->HB = NPU_MALLOC(h_size);
        NPU_ASSERT(hnd->HB != NULL);
        if (layer->init_H_offset == -1) {
            // if init_H_offset is -1, set init hidden state to 0.
            NPU_MEMSET(hnd->HB, 0, h_size);
        } else {
            void *init_HB = (void *)((i32)model_weight + layer->init_HB_offset);
            // convert init hidden state to float
            DeQuant(init_HB, NPU_INT8, hnd->HB, NPU_FP32, layer->batch_size * layer->hidden_size, layer->q_init_HB.scale, 0);
        }
    }

    if (Gru_fullcache_check(layer)) {
        hnd->full_cache_mode = 1;
        GRU_NPU_LOG("this gru layer can be run in full cache mode\n");
    } else if (Gru_normal_mem_check(layer)) {
        GRU_NPU_LOG("this gru layer can not be run in normal mode\n");
    } else {
        NPU_ASSERT_MSG(0, "memory is not enough for this gru layer\n");
    }

    GRU_NPU_LOG("GruLayerInit finish. layer=%p, h_size=%lu\n", hnd->layer, h_size);

    return hnd;
}

i32 GruLayerFini(GruLayerHnd_t hnd)
{
    GRU_NPU_LOG("finish, hnd=%p\n", hnd);
    GRU_NPU_ASSERT(hnd != NULL);

    // release hidden state
    if (hnd->H)
        NPU_FREE(hnd->H);
    if (hnd->HB)
        NPU_FREE(hnd->HB);

    NPU_FREE(hnd);
    return 0;
}

/**
 * @brief gru layer process function
 *
 * equation:
 * $$ z_t = \sigma(X_t * W_z + H_{t-1} * R_z + W_{bz} + R_{bz}) $$
 * $$ r_t = \sigma(X_t * W_r + H_{t-1} * R_r + W_{br} + R_{br}) $$
 * $$ \hat{h_t} = tanh(X_t * W_h + (r_t \odot H_{t-1}) * R_h + W_{bh} + R_{bh}) $$
 * $$ h_t = (1 - z_t) \odot \hat{h_t} + z_t \odot H_{t-1} $$
 *
 * run flow:
 * 1. load Xt, if quantization is needed, quantize it.
 * 2. load Wr and Wbr
 * 3. load Ht-1, Rr and Rbr, calculate Xt * Wr + Wbr
 * 4. quantize Ht-1
 * 5. load Wz and Wbz, calculate Ht-1 * Rz + Rbz
 *
 *
 * @param param param[0](GruLayerHnd_t): gru layer handle, param[1](void*): input data pointer, param[2](void*): output data pointer
 */
static void gru_process(u32 *param)
{
    GRU_NPU_LOG("start gru_process in normal mode\n");
    GruLayerHnd_t hnd = (GruLayerHnd_t)(((void **)param)[0]);
    void         *in  = ((void **)param)[1];
    void         *out = ((void **)param)[2];
    GRU_NPU_LOG("enter %s, hnd: %p, in: %p, out: %p\n", __func__, hnd, in, out);

    GruLayer_t *layer = hnd->layer;
    NPU_ASSERT((layer->pre.quant == QUANT_NONE) || (layer->pre.quant == QUANT_QUANT));

    LayerQuant_t input_quant = layer->pre.quant;
    i32          seq_length  = layer->seq_length;
    i32          batch_size  = layer->batch_size;
    i32          input_size  = layer->input_size;
    i32          hidden_size = layer->hidden_size;

    // length of input.
    u32 Xt_len = layer->batch_size * layer->input_size;
    // size of input.
    u32 Xt_input_size = Xt_len * NPU_TYPE_WIDTH(input_quant == QUANT_NONE ? layer->i_type : layer->pre.qi_type);
    // size of Xt.
    u32 Xt_size = Xt_len * NPU_TYPE_WIDTH(layer->i_type);
    // weights size for paramater each gate.
    u32 w_size = layer->hidden_size * layer->input_size * NPU_TYPE_WIDTH(layer->w_type);
    // weights size for recurrent each gate.
    u32 r_size = layer->hidden_size * layer->hidden_size * NPU_TYPE_WIDTH(layer->w_type);
    // bias size for each gate.
    u32 b_size = layer->hidden_size * NPU_TYPE_WIDTH(layer->b_type);
    // hidden state length.
    u32 h_len = layer->batch_size * layer->hidden_size;
    // hidden state float size.
    u32 h_fp_size = h_len * NPU_TYPE_WIDTH(NPU_FP32);
    // hidden state quant size.
    u32 h_quant_size = h_len * NPU_TYPE_WIDTH(NPU_INT8);
    // size of output.
    u32 output_size = layer->post.quant == QUANT_NONE ? h_fp_size : h_quant_size;

    u32 l1_start_used = 0;
    u32 l1_end_used   = 0;
    u32 l1_start      = L1_CNN_Base;
    u32 l1_end        = L1_CNN_Base + L1_CNN_Size;
    u32 H_fp          = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_fp_size);
    u32 H_quant = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_quant_size);
    // u32 zt = l1_start + l1_start_used;
    // l1_start_used += NPU_ALIGN_64(h_fp_size);
    u32 rt = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_fp_size);
    u32 ht = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_fp_size);
    u32 Wzrh = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(w_size);
    u32 Rzrh = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(w_size);
    u32 Wbzrh = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(b_size);
    u32 Rbzrh = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(b_size);
    u32 Xt = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(Xt_size);
    GRU_NPU_ASSERT(l1_start_used + l1_end_used <= L1_CNN_Size);
    GRU_NPU_LOG("l1_start_used=%lu, l1_end_used=%lu, free=%lu\n",
                l1_start_used, l1_end_used, l1_end - l1_start - l1_start_used - l1_end_used);
    GRU_NPU_LOG("l1_start: %p, l1_end: %p\n", (void *)l1_start, (void *)l1_end);

    // init tsme to do matmul
    SUBMIT_TASK(init_tsme, MATMUL_CORE, 0, NULL, 0);

    Task_status *quant_H_status     = NULL;
    Task_status *input_status       = NULL;
    Task_status *output_status      = NULL;
    Task_status *tsme_config_status = NULL;

    Task_status *matmul_status = NULL;
    Task_status *fp_status     = NULL;

    void *model_W;
    void *model_R;
    void *model_Wb;
    void *model_Rb;
    void *pre_H;
    // quant input,h,w,r scale pointer
    float *p_Xscale = &layer->q_i.scale;
    float *p_Hscale;
    float *p_Wscale;
    float *p_Rscale;

    GRU_NPU_LOG("layer->direction %d\n", layer->direction);
    for (GruDirection direction = GRU_FORWARD; direction <= GRU_BIDIRECTIONAL; direction <<= 1) {
        if (!(direction & layer->direction)) {
            continue;
        }

        GRU_NPU_LOG("start direction %d\n", direction);
        int32_t input_shift_size  = Xt_input_size;
        int32_t output_shift_size = output_size;
        // set address of weights and bias.
        if (direction == GRU_FORWARD) {
            model_W  = hnd->W;
            model_R  = hnd->R;
            model_Wb = hnd->Wb;
            model_Rb = hnd->Rb;
            pre_H    = hnd->H;

            p_Hscale = &layer->q_h.scale;
            p_Wscale = &layer->q_w.scale;
            p_Rscale = &layer->q_r.scale;
        } else {
            model_W  = hnd->WB;
            model_R  = hnd->RB;
            model_Wb = hnd->WBb;
            model_Rb = hnd->RBb;
            pre_H    = hnd->HB;

            p_Hscale = &layer->q_hb.scale;
            p_Wscale = &layer->q_wb.scale;
            p_Rscale = &layer->q_rb.scale;

            // move input to backward input
            in = (void *)((i8 *)in + input_shift_size * (seq_length - 1));
            // move output to backward output
            out = (void *)((i8 *)out + output_shift_size * (seq_length - 1));

            input_shift_size  = -input_shift_size;
            output_shift_size = -output_shift_size;
        }

        // if CONCAT OUTPUT, output shift size require multiply 2.
        if (layer->direction == GRU_BIDIRECTIONAL && layer->bidirect_concat) {
            output_shift_size *= 2;
        }

        // load Ht-1
        Task_status *load_H_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)H_fp, (u32)pre_H, (u32)h_fp_size, (u32)DMA_L2_TO_L1);
        for (i32 step = 0; step < seq_length; step++) {
            GRU_NPU_LOG("start step %d\n", step);

            GRU_NPU_LOG("quant Ht-1\n");
            // quantize Ht-1
            if (step != 0) {
                quant_H_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, NULL, 5, (u32)H_fp, (u32)H_quant, (u32)h_len, (u32)(p_Hscale), (u32)layer->dynamic_q_h);
            } else {
                quant_H_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, load_H_status, 5, (u32)H_fp, (u32)H_quant, (u32)h_len, (u32)(p_Hscale), (u32)layer->dynamic_q_h);
                load_H_status  = NULL;
            }
            // GRU_NPU_LOG("H_quant[0]=%d, H_quant[1]=%d, H_quant[2]=%d\n", ((i8 *)H_quant)[0], ((i8 *)H_quant)[1], ((i8 *)H_quant)[2]);

            // load Rr and Rbr
            SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Rzrh, (u32)model_R + w_size, (u32)w_size, (u32)DMA_L2_TO_L1);
            Task_status *load_r_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Rbzrh, (u32)model_Rb + b_size, (u32)b_size, (u32)DMA_L2_TO_L1);

            GRU_NPU_LOG("prepare input data\n");
            Task_status *Xt_dma_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Xt, (u32)in + step * input_shift_size, (u32)Xt_input_size, (u32)DMA_L2_TO_L1);
            GRU_NPU_LOG("Xt[0]*10000: %d, Xt[1]*10000: %d, Xt[2]*10000: %d\n", (i32)(((float *)Xt)[0] * 10000), (i32)(((float *)Xt)[1] * 10000), (i32)(((float *)Xt)[2] * 10000));

            // cal Ht-1 * (Rr^T) + Rbr, save to rt
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)hidden_size, (u32)hidden_size);
            WAIT_TASK_FINISH(quant_H_status);
            matmul_status = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, load_r_status, 7, (u32)H_quant, (u32)Rzrh, (u32)Rbzrh, (u32)rt, (u32)layer->batch_size, (u32)layer->hidden_size, (u32)layer->hidden_size);
            // convert rt from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)rt, (u32)h_len, *(u32 *)p_Hscale, *(u32 *)p_Rscale);

            if (input_quant == QUANT_QUANT) {
                /* quantization input */
                GRU_NPU_LOG("quant Xt\n");
                // convert input data from fp32 to int8
                input_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, Xt_dma_status, 5, (u32)Xt, (u32)Xt, (u32)Xt_len, (u32)(p_Xscale), (u32)layer->dynamic_q_i);
                GRU_NPU_LOG("quant Xt[0]: %d, Xt[1]: %d, Xt[2]: %d\n", ((i8 *)Xt)[0], ((i8 *)Xt)[1], ((i8 *)Xt)[2]);
            } else {
                input_status = Xt_dma_status;
            }
            GRU_NPU_LOG("prepare input data done!\n");

            // load Wr and Wbr
            SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Wzrh, (u32)model_W + w_size, (u32)w_size, (u32)DMA_L2_TO_L1);
            Task_status *load_w_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Wbzrh, (u32)model_Wb + b_size, (u32)b_size, (u32)DMA_L2_TO_L1);
            GRU_NPU_LOG("load Wr and Wbr done!\n");

            // cal Xt * (Wr^T) + Wbr, save to ht
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)input_size, (u32)hidden_size);
            WAIT_TASK_FINISH(load_w_status);
            matmul_status = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, input_status, 7, (u32)Xt, (u32)Wzrh, (u32)Wbzrh, (u32)ht, (u32)layer->batch_size, (u32)layer->input_size, (u32)layer->hidden_size);
            // convert Xt_Wr from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)ht, (u32)h_len, *(u32 *)p_Xscale, *(u32 *)p_Wscale);
            // GRU_NPU_LOG("Xt_Wr[0]: %f, Xt_Wr[1]: %f, Xt_Wr[2]: %f\n", ((float *)ht)[0], ((float *)ht)[1], ((float *)ht)[2]);
            GRU_NPU_LOG("cal Xt * (Wr^T) + Wbr done!\n");

            // add Xt * (Wr^T) + Wbr and Ht-1 * (Rr^T) + Rbr, save to rt
            Task_status *ht_release_status = SUBMIT_TASK(fp_element_wise_add, FP_CORE, 0, NULL, 3, (u32)rt, (u32)ht, (u32)h_len);
            // activation for rt
            SUBMIT_TASK(fp_sigmoid, FP_CORE, 0, NULL, 2, (u32)rt, (u32)h_len);
            // GRU_NPU_LOG("rt[0]=%f, rt[1]=%f, rt[2]=%f\n", ((float *)rt)[0], ((float *)rt)[1], ((float *)rt)[2]);
            GRU_NPU_LOG("rt[0]*10000=%d, rt[1]*10000=%d, rt[2]*10000=%d\n", (int)(((float *)rt)[0] * 10000), (int)(((float *)rt)[1] * 10000), (int)(((float *)rt)[2] * 10000));
            GRU_NPU_LOG("cal rt done!\n");

            // cal rt.Ht-1, save to rt
            Task_status *H_fp_release_status = SUBMIT_TASK(fp_element_wise_mul, FP_CORE, 0, NULL, 3, (u32)rt, (u32)H_fp, (u32)h_len);
            if (step != 0) {
                // save H_fp to L2, release H_fp
                H_fp_release_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, H_fp_release_status, 4, (u32)H_fp, (u32)pre_H, (u32)h_fp_size, (u32)DMA_L1_TO_L2);
            }
            // quantize rt.Ht-1
            u32 rt_ht_scale;
            if (layer->dynamic_q_h)
            {
                fp_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, NULL, 5, (u32)rt, (u32)rt, (u32)h_len, (u32)(&rt_ht_scale), (u32)1);
            } else {
                memcpy(&rt_ht_scale, p_Hscale, sizeof(float));
                fp_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, NULL, 5, (u32)rt, (u32)rt, (u32)h_len, (u32)(&rt_ht_scale), (u32)0);
            }

            // load Wh and Wbh
            SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Wzrh, (u32)model_W + 2 * w_size, (u32)w_size, (u32)DMA_L2_TO_L1);
            load_w_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Wbzrh, (u32)model_Wb + 2 * b_size, (u32)b_size, (u32)DMA_L2_TO_L1);
            // load Rh and Rhb
            SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Rzrh, (u32)model_R + 2 * w_size, (u32)w_size, (u32)DMA_L2_TO_L1);
            load_r_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Rbzrh, (u32)model_Rb + 2 * b_size, (u32)b_size, (u32)DMA_L2_TO_L1);

            // cal Xt * (Wh^T) + Wbh, save to ht
            WAIT_TASK_FINISH(ht_release_status);
            matmul_status = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, load_w_status, 7, (u32)Xt, (u32)Wzrh, (u32)Wbzrh, (u32)ht, (u32)layer->batch_size, (u32)layer->input_size, (u32)layer->hidden_size);
            // convert ht(Xt * (Wh^T) + Wbh) from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)ht, (u32)h_len, *(u32 *)p_Xscale, *(u32 *)p_Wscale);
            GRU_NPU_LOG("cal Xt * (Wh^T) + Wbh done!\n");

            // cal (rt.Ht-1) * (Rh^T) + Rhb, save to H_fp
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)hidden_size, (u32)hidden_size);
            matmul_status      = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, fp_status, 7, (u32)rt, (u32)Rzrh, (u32)Rbzrh, (u32)H_fp, (u32)layer->batch_size, (u32)layer->hidden_size, (u32)layer->hidden_size);
            // convert from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, NULL, 4, (u32)H_fp, (u32)h_len, rt_ht_scale, *(u32 *)p_Rscale);
            // add (Xt * (Wh^T) + Wbh) and ((rt.Ht-1) * (Rh^T) + Rhb), save to ht
            H_fp_release_status = SUBMIT_TASK(fp_element_wise_add, FP_CORE, 0, NULL, 3, (u32)ht, (u32)H_fp, (u32)h_len);
            // activation for ht
            SUBMIT_TASK(fp_tanh, FP_CORE, 0, NULL, 2, (u32)ht, (u32)h_len);
            // GRU_NPU_LOG("ht[0]: %f, ht[1]: %f, ht[2]: %f\n", ((float*)ht)[0], ((float *)ht)[1], ((float *)ht)[2]);
            GRU_NPU_LOG("ht[0]*10000=%d, ht[1]*10000=%d, ht[2]*10000=%d\n", (int)(((float *)ht)[0] * 10000), (int)(((float *)ht)[1] * 10000), (int)(((float *)ht)[2] * 10000));
            GRU_NPU_LOG("cal ht done!\n");

            // load Wz and Wbz
            SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Wzrh, (u32)model_W, (u32)w_size, (u32)DMA_L2_TO_L1);
            load_w_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Wbzrh, (u32)model_Wb, (u32)b_size, (u32)DMA_L2_TO_L1);
            // load Rz and Rbz
            SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Rzrh, (u32)model_R, (u32)w_size, (u32)DMA_L2_TO_L1);
            load_r_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Rbzrh, (u32)model_Rb, (u32)b_size, (u32)DMA_L2_TO_L1);

            // wait rt release
            WAIT_TASK_FINISH(matmul_status);
            // rt is released, use it to save zt
            u32 zt = rt;
            // cal Xt * (Wz^T) + Wbz, save to zt
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)input_size, (u32)hidden_size);
            matmul_status      = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, load_w_status, 7, (u32)Xt, (u32)Wzrh, (u32)Wbzrh, (u32)zt, (u32)layer->batch_size, (u32)layer->input_size, (u32)layer->hidden_size);
            // convert from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)zt, (u32)h_len, *(u32 *)p_Xscale, *(u32 *)p_Wscale);
            GRU_NPU_LOG("cal Xt * (Wz^T) + Wbz done!\n");

            // reload H_fp (overwrited before)
            load_H_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, H_fp_release_status, 4, (u32)H_fp, (u32)pre_H, (u32)h_fp_size, (u32)DMA_L2_TO_L1);

            // cal Ht-1 * (Rz^T) + Rbz, save to Wzrh
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)hidden_size, (u32)hidden_size);
            matmul_status      = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, load_r_status, 7, (u32)H_quant, (u32)Rzrh, (u32)Rbzrh, (u32)Wzrh, (u32)layer->batch_size, (u32)layer->hidden_size, (u32)layer->hidden_size);
            GRU_NPU_LOG("cal Ht-1 * (Rz^T) + Rbz done!\n");
            // convert from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)Wzrh, (u32)h_len, *(u32 *)p_Hscale, *(u32 *)p_Rscale);
            // add (Xt * (Wz^T) + Wbz) and (Ht-1 * (Rz^T) + Rbz), save to zt
            SUBMIT_TASK(fp_element_wise_add, FP_CORE, 0, NULL, 3, (u32)zt, (u32)Wzrh, (u32)h_len);
            // activation for zt
            SUBMIT_TASK(fp_sigmoid, FP_CORE, 0, NULL, 2, (u32)zt, (u32)h_len);
            // GRU_NPU_LOG("zt[0]: %f, zt[1]: %f, zt[2]: %f\n", ((float*)zt)[0], ((float *)zt)[1], ((float *)zt)[2]);
            GRU_NPU_LOG("zt[0]*10000=%d, zt[1]*10000=%d, zt[2]*10000=%d\n", (int)(((float *)zt)[0] * 10000), (int)(((float *)zt)[1] * 10000), (int)(((float *)zt)[2] * 10000));
            GRU_NPU_LOG("cal zt done!\n");

            // cal new hidden state H_fp
            fp_status = SUBMIT_TASK(gru_update_hidden_state, FP_CORE, 0, NULL, 4, (u32)H_fp, (u32)zt, (u32)ht, (u32)h_len);
            // GRU_NPU_LOG("Ht[0]: %f, Ht[1]: %f, Ht[2]: %f\n", ((float *)H_fp)[0], ((float *)H_fp)[1], ((float *)H_fp)[2]);
            GRU_NPU_LOG("Ht[0]*10000=%d, Ht[1]*10000=%d, Ht[2]*10000=%d\n", (int)(((float *)H_fp)[0] * 10000), (int)(((float *)H_fp)[1] * 10000), (int)(((float *)H_fp)[2] * 10000));
            GRU_NPU_LOG("cal Ht done!\n");

            // copy output
            if (layer->post.quant == QUANT_NONE) {
                GRU_NPU_LOG("copy output data from l1 to l2 without quantization\n");
                output_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, fp_status, 4, (u32)H_fp, (u32)out + step * output_shift_size, (u32)h_fp_size, (u32)DMA_L1_TO_L2);
            } else if (layer->post.quant == QUANT_FSCALE) {
                GRU_NPU_LOG("copy output data from l1 to l2 with quantization\n");
                // quant H_fp, save to H_quant
                quant_H_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, fp_status, 5, (u32)H_fp, (u32)H_quant, (u32)h_len, (u32)(&layer->post.quant_u.q_fp.scale), (u32)layer->dynamic_q_o);
                output_status  = SUBMIT_TASK(dma_1d, DMA_CORE, 0, quant_H_status, 4, (u32)H_quant, (u32)out + step * output_shift_size, (u32)h_quant_size, (u32)DMA_L1_TO_L2);
            } else {
                GRU_NPU_ASSERT((layer->post.quant == QUANT_NONE) || (layer->post.quant == QUANT_FSCALE));
            }
            GRU_NPU_LOG("copy output done!\n");
        }
        // save new hidden state
        output_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)H_fp, (u32)pre_H, (u32)h_fp_size, (u32)DMA_L1_TO_L2);
    }

    WAIT_TASK_FINISH(output_status);

    // all done
    task_queue_exit = 1;
    GRU_NPU_LOG("gru_process done!\n");
}

/**
 * @brief gru layer process function
 *
 * equation:
 * $$ z_t = \sigma(X_t * W_z + H_{t-1} * R_z + W_{bz} + R_{bz}) $$
 * $$ r_t = \sigma(X_t * W_r + H_{t-1} * R_r + W_{br} + R_{br}) $$
 * $$ \hat{h_t} = tanh(X_t * W_h + (r_t \odot H_{t-1}) * R_h + W_{bh} + R_{bh}) $$
 * $$ h_t = (1 - z_t) \odot \hat{h_t} + z_t \odot H_{t-1} $$
 *
 * run flow:
 * 1. load Xt, if quantization is needed, quantize it.
 * 2. load Wr and Wbr
 * 3. load Ht-1, Rr and Rbr, calculate Xt * Wr + Wbr
 * 4. quantize Ht-1
 * 5. load Wz and Wbz, calculate Ht-1 * Rz + Rbz
 *
 *
 * @param param param[0](GruLayerHnd_t): gru layer handle, param[1](void*): input data pointer, param[2](void*): output data pointer, param[3](GruDirection): gru direction
 */
static void gru_process_with_fullcache(u32 *param)
{
    GRU_NPU_LOG("start gru_process in full cache mode\n");
    GruLayerHnd_t hnd = (GruLayerHnd_t)(((void **)param)[0]);
    void         *in  = ((void **)param)[1];
    void         *out = ((void **)param)[2];
    GRU_NPU_LOG("enter %s, hnd: %p, in: %p, out: %p\n", __func__, hnd, in, out);

    GruLayer_t *layer = hnd->layer;
    NPU_ASSERT((layer->pre.quant == QUANT_NONE) || (layer->pre.quant == QUANT_QUANT));

    LayerQuant_t input_quant = layer->pre.quant;
    i32          seq_length  = layer->seq_length;
    i32          batch_size  = layer->batch_size;
    i32          input_size  = layer->input_size;
    i32          hidden_size = layer->hidden_size;

    // length of Xt.
    u32 Xt_len = batch_size * input_size;
    // size of input.
    u32 Xt_input_size = Xt_len * NPU_TYPE_WIDTH(input_quant == QUANT_NONE ? layer->i_type : layer->pre.qi_type);
    // size of Xt.
    u32 Xt_size = Xt_len * NPU_TYPE_WIDTH(layer->i_type);
    // length of Wz/Wr/Wh
    u32 w_len = input_size * hidden_size;
    // weights size for paramater each gate.
    u32 w_size = w_len * NPU_TYPE_WIDTH(layer->w_type);
    // parameter weights size
    u32 W_size = w_size * 3;
    // length of Rz/Rr/Rh
    u32 r_len = hidden_size * hidden_size;
    // weights size for recurrent each gate.
    u32 r_size = r_len * NPU_TYPE_WIDTH(layer->w_type);
    // recurrence weights size
    u32 R_size = r_size * 3;
    // bias size for each gate.
    u32 b_size = hidden_size * NPU_TYPE_WIDTH(layer->b_type);
    // parameter/recurrence gate bias size
    u32 B_size = b_size * 3;
    // hidden state length.
    u32 h_len = batch_size * hidden_size;
    // hidden state float size.
    u32 h_fp_size = h_len * NPU_TYPE_WIDTH(NPU_FP32);
    // hidden state quant size.
    u32 h_quant_size = h_len * NPU_TYPE_WIDTH(NPU_INT8);
    // size of output.
    u32 output_size = layer->post.quant == QUANT_NONE ? h_fp_size : h_quant_size;

    u32 l1_start_used = 0;
    u32 l1_end_used   = 0;
    u32 l1_start      = L1_CNN_Base;
    u32 l1_end        = L1_CNN_Base + L1_CNN_Size;
    u32 H_fp          = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_fp_size);
    u32 H_quant = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_quant_size);
    u32 zt = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_fp_size);
    u32 rt = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_fp_size);
    u32 ht = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(h_fp_size);
    u32 W = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(W_size);
    u32 R = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(R_size);
    u32 Wb = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(B_size);
    u32 Rb = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(B_size);
    u32 Xt = l1_start + l1_start_used;
    l1_start_used += NPU_ALIGN_64(Xt_input_size);
    GRU_NPU_ASSERT(l1_start_used + l1_end_used <= L1_CNN_Size);
    GRU_NPU_LOG("l1_start_used=%lu, l1_end_used=%lu, free=%lu\n",
                l1_start_used, l1_end_used, l1_end - l1_start - l1_start_used - l1_end_used);
    GRU_NPU_LOG("l1_start: %p, l1_end: %p\n", (void *)l1_start, (void *)l1_end);

    // init tsme to do matmul
    SUBMIT_TASK(init_tsme, MATMUL_CORE, 0, NULL, 0);

    Task_status *quant_H_status     = NULL;
    Task_status *input_status       = NULL;
    Task_status *output_status      = NULL;
    Task_status *tsme_config_status = NULL;

    Task_status *matmul_status = NULL;
    Task_status *fp_status     = NULL;

    void *model_W;
    void *model_R;
    void *model_Wb;
    void *model_Rb;
    void *pre_H;
    // quant input,h,w,r scale pointer
    float *p_Xscale = &layer->q_i.scale;
    float *p_Hscale;
    float *p_Wscale;
    float *p_Rscale;

    GRU_NPU_LOG("layer->direction %d\n", layer->direction);
    for (GruDirection direction = GRU_FORWARD; direction <= GRU_BIDIRECTIONAL; direction <<= 1) {
        if (!(direction & layer->direction)) {
            continue;
        }

        GRU_NPU_LOG("start direction %d\n", direction);
        int32_t input_shift_size  = Xt_input_size;
        int32_t output_shift_size = output_size;
        // set address of weights and bias.
        if (direction == GRU_FORWARD) {
            model_W  = hnd->W;
            model_R  = hnd->R;
            model_Wb = hnd->Wb;
            model_Rb = hnd->Rb;
            pre_H    = hnd->H;

            p_Hscale = &layer->q_h.scale;
            p_Wscale = &layer->q_w.scale;
            p_Rscale = &layer->q_r.scale;
        } else {
            model_W  = hnd->WB;
            model_R  = hnd->RB;
            model_Wb = hnd->WBb;
            model_Rb = hnd->RBb;
            pre_H    = hnd->HB;

            p_Hscale = &layer->q_hb.scale;
            p_Wscale = &layer->q_wb.scale;
            p_Rscale = &layer->q_rb.scale;

            // move input to backward input
            in = (void *)((i8 *)in + input_shift_size * (seq_length - 1));
            // move output to backward output
            out = (void *)((i8 *)out + output_shift_size * (seq_length - 1));

            input_shift_size  = -input_shift_size;
            output_shift_size = -output_shift_size;
        }

        // if CONCAT OUTPUT, output shift size require multiply 2.
        if (layer->direction == GRU_BIDIRECTIONAL && layer->bidirect_concat) {
            output_shift_size *= 2;
        }

        // load Ht-1
        Task_status *load_H_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)H_fp, (u32)pre_H, (u32)h_fp_size, (u32)DMA_L2_TO_L1);
        // load R, Rb
        SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)R, (u32)model_R, (u32)R_size, (u32)DMA_L2_TO_L1);
        Task_status *load_R_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Rb, (u32)model_Rb, (u32)B_size, (u32)DMA_L2_TO_L1);
        // load W, Wb
        SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)W, (u32)model_W, (u32)W_size, (u32)DMA_L2_TO_L1);
        Task_status *load_W_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Wb, (u32)model_Wb, (u32)B_size, (u32)DMA_L2_TO_L1);

        for (i32 step = 0; step < seq_length; step++) {
            GRU_NPU_LOG("start step %d\n", step);

            GRU_NPU_LOG("quant Ht-1\n");
            // quantize Ht-1
            if (step != 0) {
                quant_H_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, quant_H_status, 5, (u32)H_fp, (u32)H_quant, (u32)h_len, (u32)(p_Hscale), (u32)layer->dynamic_q_h);
            } else {
                quant_H_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, load_H_status, 5, (u32)H_fp, (u32)H_quant, (u32)h_len, (u32)(p_Hscale), (u32)layer->dynamic_q_h);
                load_H_status  = NULL;
            }
            GRU_NPU_LOG("H_quant[0]=%d, H_quant[1]=%d, H_quant[2]=%d\n", ((i8 *)H_quant)[0], ((i8 *)H_quant)[1], ((i8 *)H_quant)[2]);

            GRU_NPU_LOG("prepare input data\n");
            Task_status *Xt_dma_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)Xt, (u32)in + step * input_shift_size, (u32)Xt_input_size, (u32)DMA_L2_TO_L1);
            GRU_NPU_LOG("Xt[0]*10000: %d, Xt[1]*10000: %d, Xt[2]*10000: %d\n", (i32)(((float *)Xt)[0] * 10000), (i32)(((float *)Xt)[1] * 10000), (i32)(((float *)Xt)[2] * 10000));
            // cal Ht-1 * (Rr^T) + Rbr, save to rt
            GRU_NPU_LOG("Rr[0]: %d, Rr[1]: %d, Rr[2]: %d\n", ((i8 *)(R + r_size))[0], ((i8 *)(R + r_size))[1], ((i8 *)(R + r_size))[2]);
            GRU_NPU_LOG("Rbr[0]: %d, Rbr[1]: %d, Rbr[2]: %d\n", ((i8 *)(Rb + b_size))[0], ((i8 *)(Rb + b_size))[1], ((i8 *)(Rb + b_size))[2]);
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)hidden_size, (u32)hidden_size);
            if (step != 0) {
                matmul_status = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, NULL, 7, (u32)H_quant, (u32)R + r_size, (u32)Rb + b_size, (u32)rt, (u32)batch_size, (u32)hidden_size, (u32)hidden_size);
            } else {
                while (*load_R_status != TASK_FINISH)
                    GRU_WAIT();
                load_R_status  = NULL;
                matmul_status  = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, quant_H_status, 7, (u32)H_quant, (u32)R + r_size, (u32)Rb + b_size, (u32)rt, (u32)batch_size, (u32)hidden_size, (u32)hidden_size);
                quant_H_status = NULL;
            }
            GRU_NPU_LOG("i32 Ht_Rr[0]: %d, Ht_Rr[1]: %d, Ht_Rr[2]: %d\n", ((i32 *)rt)[0], ((i32 *)rt)[1], ((i32 *)rt)[2]);
            // convert rt from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)rt, (u32)h_len, *(u32 *)p_Hscale, *(u32 *)p_Rscale);
            GRU_NPU_LOG("cal Ht-1 * (Rr^T) + Rbr done!\n");

            if (input_quant == QUANT_QUANT) {
                /* quantization input */
                GRU_NPU_LOG("quant Xt\n");
                // convert input data from fp32 to int8
                input_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, Xt_dma_status, 5, (u32)Xt, (u32)Xt, (u32)Xt_len, (u32)(&layer->q_i.scale), (u32)layer->dynamic_q_i);
                GRU_NPU_LOG("quant Xt[0]: %d, Xt[1]: %d, Xt[2]: %d\n", ((i8 *)Xt)[0], ((i8 *)Xt)[1], ((i8 *)Xt)[2]);
            } else {
                input_status = Xt_dma_status;
            }
            GRU_NPU_LOG("prepare input data done!\n");

            // cal Xt * (Wr^T) + Wbr, save to zt
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)input_size, (u32)hidden_size);
            if (step == 0) {
                while (*load_W_status != TASK_FINISH)
                    GRU_WAIT();
                load_W_status = NULL;
            }
            GRU_NPU_LOG("Wr[0]: %d, Wr[1]: %d, Wr[2]: %d\n", ((i8 *)(W + w_size))[0], ((i8 *)(W + w_size))[1], ((i8 *)(W + w_size))[2]);
            GRU_NPU_LOG("Wbr[0]: %d, Wbr[1]: %d, Wbr[2]: %d\n", ((i8 *)(Wb + b_size))[0], ((i8 *)(Wb + b_size))[1], ((i8 *)(Wb + b_size))[2]);
            matmul_status = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, input_status, 7, (u32)Xt, (u32)W + w_size, (u32)Wb + b_size, (u32)zt, (u32)batch_size, (u32)input_size, (u32)hidden_size);
            GRU_NPU_LOG("i32 Xt_Wr[0]: %d, Xt_Wr[1]: %d, Xt_Wr[2]: %d\n", ((i32 *)zt)[0], ((i32 *)zt)[1], ((i32 *)zt)[2]);
            // convert Xt * (Wr^T) + Wbr from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)zt, (u32)h_len, *(u32 *)p_Xscale, *(u32 *)p_Wscale);
            // GRU_NPU_LOG("Xt_Wr[0]: %f, Xt_Wr[1]: %f, Xt_Wr[2]: %f\n", ((float *)zt)[0], ((float *)zt)[1], ((float *)zt)[2]);
            GRU_NPU_LOG("Xt_Wr[0]*10000: %d, Xt_Wr[1]*10000: %d, Xt_Wr[2]*10000: %d\n", (i32)(((float *)zt)[0] * 10000), (i32)(((float *)zt)[1] * 10000), (i32)(((float *)zt)[2] * 10000));
            GRU_NPU_LOG("cal Xt * (Wr^T) + Wbr done!\n");

            // add Xt * (Wr^T) + Wbr and Ht-1 * (Rr^T) + Rbr, save to rt
            SUBMIT_TASK(fp_element_wise_add, FP_CORE, 0, NULL, 3, (u32)rt, (u32)zt, (u32)h_len);
            // activation for rt
            SUBMIT_TASK(fp_sigmoid, FP_CORE, 0, NULL, 2, (u32)rt, (u32)h_len);
            // GRU_NPU_LOG("rt[0]=%f, rt[1]=%f, rt[2]=%f\n", ((float *)rt)[0], ((float *)rt)[1], ((float *)rt)[2]);
            GRU_NPU_LOG("rt[0]*10000=%d, rt[1]*10000=%d, rt[2]*10000=%d\n", (int)(((float *)rt)[0] * 10000), (int)(((float *)rt)[1] * 10000), (int)(((float *)rt)[2] * 10000));
            GRU_NPU_LOG("cal rt done!\n");

            // cal rt.Ht-1, save to rt
            SUBMIT_TASK(fp_element_wise_mul, FP_CORE, 0, NULL, 3, (u32)rt, (u32)H_fp, (u32)h_len);
            // quantize rt.Ht-1
            u32 rt_ht_scale;
            if (layer->dynamic_q_h)
            {
                fp_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, NULL, 5, (u32)rt, (u32)rt, (u32)h_len, (u32)(&rt_ht_scale), (u32)1);
            } else {
                memcpy(&rt_ht_scale, p_Hscale, sizeof(float));
                fp_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, NULL, 5, (u32)rt, (u32)rt, (u32)h_len, (u32)(&rt_ht_scale), (u32)0);
            }

            // cal Xt * (Wh^T) + Wbh, save to ht
            matmul_status = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, NULL, 7, (u32)Xt, (u32)W + 2 * w_size, (u32)Wb + 2 * b_size, (u32)ht, (u32)layer->batch_size, (u32)layer->input_size, (u32)layer->hidden_size);
            // convert Xt * (Wh^T) + Wbh from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)ht, (u32)h_len, *(u32 *)p_Xscale, *(u32 *)p_Wscale);
            GRU_NPU_LOG("cal Xt * (Wh^T) + Wbh done!\n");

            // cal (rt.Ht-1) * (Rh^T) + Rhb, save to zt
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)hidden_size, (u32)hidden_size);
            matmul_status      = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, fp_status, 7, (u32)rt, (u32)R + 2 * r_size, (u32)Rb + 2 * b_size, (u32)zt, (u32)layer->batch_size, (u32)layer->hidden_size, (u32)layer->hidden_size);
            // convert from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)zt, (u32)h_len, rt_ht_scale, *(u32 *)p_Rscale);
            // add (Xt * (Wh^T) + Wbh) and ((rt.Ht-1) * (Rh^T) + Rhb), save to ht
            Task_status *zt_release_status = SUBMIT_TASK(fp_element_wise_add, FP_CORE, 0, NULL, 3, (u32)ht, (u32)zt, (u32)h_len);
            // activation for ht
            SUBMIT_TASK(fp_tanh, FP_CORE, 0, NULL, 2, (u32)ht, (u32)h_len);
            // GRU_NPU_LOG("ht[0]: %f, ht[1]: %f, ht[2]: %f\n", ((float*)ht)[0], ((float *)ht)[1], ((float *)ht)[2]);
            GRU_NPU_LOG("ht[0]*10000=%d, ht[1]*10000=%d, ht[2]*10000=%d\n", (int)(((float *)ht)[0] * 10000), (int)(((float *)ht)[1] * 10000), (int)(((float *)ht)[2] * 10000));
            GRU_NPU_LOG("cal ht done!\n");

            // cal Xt * (Wz^T) + Wbz, save to rt
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)input_size, (u32)hidden_size);
            matmul_status      = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, NULL, 7, (u32)Xt, (u32)W, (u32)Wb, (u32)rt, (u32)layer->batch_size, (u32)layer->input_size, (u32)layer->hidden_size);
            // convert from int32 to fp32
            fp_status = SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)rt, (u32)h_len, *(u32 *)p_Xscale, *(u32 *)p_Wscale);
            GRU_NPU_LOG("cal Xt * (Wz^T) + Wbz done!\n");

            // cal Ht-1 * (Rz^T) + Rbz, save to zt
            tsme_config_status = SUBMIT_TASK(update_tsme_io_size, MATMUL_CORE, 0, matmul_status, 2, (u32)hidden_size, (u32)hidden_size);
            matmul_status      = SUBMIT_TASK(gru_matmul, MATMUL_CORE, 0, zt_release_status, 7, (u32)H_quant, (u32)R, (u32)Rb, (u32)zt, (u32)layer->batch_size, (u32)layer->hidden_size, (u32)layer->hidden_size);
            GRU_NPU_LOG("cal Ht-1 * (Rz^T) + Rbz done!\n");
            // convert from int32 to fp32
            SUBMIT_TASK(int32_to_fp32, FP_CORE, 0, matmul_status, 4, (u32)zt, (u32)h_len, *(u32 *)p_Hscale, *(u32 *)p_Rscale);
            // add (Xt * (Wz^T) + Wbz) and (Ht-1 * (Rz^T) + Rbz), save to zt
            SUBMIT_TASK(fp_element_wise_add, FP_CORE, 0, NULL, 3, (u32)zt, (u32)rt, (u32)h_len);
            // activation for zt
            SUBMIT_TASK(fp_sigmoid, FP_CORE, 0, NULL, 2, (u32)zt, (u32)h_len);
            // GRU_NPU_LOG("zt[0]: %f, zt[1]: %f, zt[2]: %f\n", ((float*)zt)[0], ((float *)zt)[1], ((float *)zt)[2]);
            GRU_NPU_LOG("zt[0]*10000=%d, zt[1]*10000=%d, zt[2]*10000=%d\n", (int)(((float *)zt)[0] * 10000), (int)(((float *)zt)[1] * 10000), (int)(((float *)zt)[2] * 10000));
            GRU_NPU_LOG("cal zt done!\n");

            // cal new hidden state Ht
            fp_status = SUBMIT_TASK(gru_update_hidden_state, FP_CORE, 0, NULL, 4, (u32)H_fp, (u32)zt, (u32)ht, (u32)h_len);
            GRU_NPU_LOG("Ht[0]: %f, Ht[1]: %f, Ht[2]: %f\n", ((float *)H_fp)[0], ((float *)H_fp)[1], ((float *)H_fp)[2]);
            GRU_NPU_LOG("Ht[0]*10000=%d, Ht[1]*10000=%d, Ht[2]*10000=%d\n", (int)(((float *)H_fp)[0] * 10000), (int)(((float *)H_fp)[1] * 10000), (int)(((float *)H_fp)[2] * 10000));
            GRU_NPU_LOG("cal Ht done!\n");

            // copy output
            if (layer->post.quant == QUANT_NONE) {
                GRU_NPU_LOG("copy output data from l1 to l2 without quantization\n");
                output_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, fp_status, 4, (u32)H_fp, (u32)out + step * output_shift_size, (u32)h_fp_size, (u32)DMA_L1_TO_L2);
            } else if (layer->post.quant == QUANT_FSCALE) {
                GRU_NPU_LOG("copy output data from l1 to l2 with quantization\n");
                // quant H_fp, save to H_quant
                quant_H_status = SUBMIT_TASK(fp32_to_int8, FP_CORE, 0, fp_status, 5, (u32)H_fp, (u32)H_quant, (u32)h_len, (u32)(&layer->post.quant_u.q_fp.scale), (u32)layer->dynamic_q_o);
                output_status  = SUBMIT_TASK(dma_1d, DMA_CORE, 0, quant_H_status, 4, (u32)H_quant, (u32)out + step * output_shift_size, (u32)h_quant_size, (u32)DMA_L1_TO_L2);
            } else {
                GRU_NPU_ASSERT((layer->post.quant == QUANT_NONE) || (layer->post.quant == QUANT_FSCALE));
            }
            GRU_NPU_LOG("copy output done!\n");
        }
        // save new hidden state
        output_status = SUBMIT_TASK(dma_1d, DMA_CORE, 0, NULL, 4, (u32)H_fp, (u32)pre_H, (u32)h_fp_size, (u32)DMA_L1_TO_L2);
    }

    if (*output_status != TASK_FINISH) {
        GRU_WAIT();
    }

    // all done
    task_queue_exit = 1;
    GRU_NPU_LOG("gru_process done!\n");
}

i32 GruLayer(GruLayerHnd_t hnd, void *in, void *weight, void *out, PerfRsp_t *rsp)
{
    NpuPerf_t perf;
    if (rsp != NULL) {
        NpuPerfInit(&perf);
    }

    GRU_NPU_ASSERT(hnd != NULL);
    GRU_NPU_ASSERT(in != NULL);
    GRU_NPU_ASSERT(weight != NULL);
    GRU_NPU_ASSERT(out != NULL);
    GRU_NPU_ASSERT(hnd->layer != NULL);
    GRU_NPU_ASSERT(hnd->layer->i_fmt == hnd->layer->o_fmt);
    GRU_NPU_ASSERT(hnd->W != NULL);

    // GRU_NPU_LOG("hnd=0x%x, layer=0x%x, in=0x%x, out=0x%x\n", hnd, hnd->layer, in, out);
    /* Move layer info into stack(CCL1) */
    GruLayerMgr_t gru_mgr;
    GruLayerHnd_t gru_hnd = &gru_mgr;
    GruLayer_t    gru_layer;
    NPU_MEMCPY(&gru_mgr, hnd, sizeof(GruLayerMgr_t));
    NPU_MEMCPY(&gru_layer, hnd->layer, sizeof(GruLayer_t));
    gru_mgr.layer = &gru_layer;
    // GRU_NPU_LOG("stack gru_hnd=0x%x, gru_layer=0x%x, in=0x%x, out=0x%x\n", gru_hnd, gru_hnd->layer, in, out);

    gru_multi_core_init();

    if (gru_mgr.full_cache_mode) {
        SUBMIT_TASK(gru_process_with_fullcache, SCHEDULE_CORE, 0, NULL, 3, (u32)gru_hnd, (u32)in, (u32)out);
    } else {
        SUBMIT_TASK(gru_process, SCHEDULE_CORE, 0, NULL, 3, (u32)gru_hnd, (u32)in, (u32)out);
    }
    if (!(SINGLE_CORE_MODE))
    {
        // TODO: I don't known why? but it faster than `CcFork((void (*)(void *))gru_multi_core, NULL);`
        void *params[4] = {gru_hnd, in, out};
        CcFork(gru_multi_core, params);
    }

    GRU_NPU_LOG("GruLayer done!\n");

    if (rsp != NULL) {
        rsp->time_us = NpuPerfGetUs(&perf);
    }
    return 0;
}

#undef LOG_TAG

#pragma GCC pop_options