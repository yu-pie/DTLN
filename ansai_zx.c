/* FreeRTOS kernel includes. */
#include "FreeRTOS.h"
#include "semphr.h"
#include "task.h"
#include "priority_list.h"

/*AT1K includes*/
#include "ts_device.h"
#include "ts_hal_device.h"
#include "ts_print.h"

#include "hci.h"
#include "host_log.h"
#include "model_verify_helper.h"
#include "ts_psram.h"
#include "fft_wrap.h"
#include <math.h>
#include "ts_perf.h"
//#include "board.h"
#include "ts_ops.h"
#include "misc.h"
#include "brb.h"
#include "ansai_zx.h"
#include "sw_fft.h"
#include "alg_module.h"
#include "psram_heap.h"

#define INT16_MAX_IN_FLOAT 32767.0f

#define TEST_PRINTF         ts_printf
#define NPU_ATTRI_ALIGN8    __attribute__ ((aligned (8)))
#define NPU_ATTRI_ALIGN64   __attribute__ ((aligned (64)))

#define CCL1_ALIGN_64(addr) ((addr + 63) & 0xFFFFFFC0)

static PERF_INIT(init);

#define PERF_PRINT(log_level, model_perf, s, per_cnt) \
    do { \
        static int i = 0; \
        \
        i++; \
        if ((i % per_cnt) == 0) { \
            TEST_PRINTF("%s", s); \
            TEST_PRINTF(" init %d, pre:%d, model:%d/%d, irfft:%d, post:%d, mcu:%d, npu: %d\n", \
                    model_perf.init_us, \
                    model_perf.pre_us, \
                    model_perf.model1_us, \
                    model_perf.model1_npu_us, \
                    model_perf.irfft_us, \
                    model_perf.post_us, \
                    model_perf.mcu_handle_us, \
                    model_perf.npu_handle_us); \
        } \
    } while (0)

#define BLOCK_LEN           (512)
#define BLOCK_SHIFT         (256)
#define BLOCK_OVERLAP       (BLOCK_LEN - BLOCK_SHIFT)
#define RFFT_LEN            (BLOCK_LEN / 2 + 1)
#define MODEL_IN_CNT        (264)
#define MODEL_OUT_CNT       (264)
#define MODEL_BUF_CNT       (5)
#define MAG_LEN             (MODEL_IN_CNT)
#define PHASE_LEN           ((RFFT_LEN) * 2)

#define NPU_HEAP_SIZE       (128*1024)
#define NPU_PRINT_BUF_SIZE  (128)

#if 0
static void _print_v_f(const char* tag, float* buf, int start, int end)
{
    TEST_PRINTF("%s", tag);
    for (int i = start; i < end; i++) {
        TEST_PRINTF("%f,", buf[i]);
    }
    TEST_PRINTF("\n");
}
#else 
#define _print_v_f
#endif

typedef struct {
    AudioCallFunc_t func;
    int fft_len;
    int* int_work_area;
    int int_work_area_len;
    float* fp_work_area;
    int fp_work_area_len;
    int heap_used;
} _SwFftMgr_t;

typedef _SwFftMgr_t* _SwFftHnd_t;

extern void Cdft(int n, int isgn, float *a, int *ip, float *w);
extern void Rdft(int n, int isgn, float *a, int *ip, float *w);
static _SwFftHnd_t _sw_fft_init(int fft_len, AudioCallFunc_t* func)
{
    if (func == NULL) {
        return NULL;
    }
    int temp = fft_len;
    if (temp < 2) {
        HAL_Printf("[FFT]", "fft_len must >= 2\n");
        return NULL;
    }
    while (temp > 1) {
        if (temp % 2 != 0) {
            HAL_Printf("[FFT]", "fft_len must be 2^n\n");
            return NULL;
        }
        temp >>= 1;
    }
    
    _SwFftMgr_t* hnd = (_SwFftMgr_t*)HAL_MALLOC(sizeof(_SwFftMgr_t));
    HAL_SANITY_CHECK(NULL != hnd);
    memset(hnd, 0, sizeof(_SwFftMgr_t));
    memcpy(&hnd->func, func, sizeof(AudioCallFunc_t));
    hnd->heap_used += sizeof(_SwFftMgr_t);

    hnd->fft_len = fft_len;
    int ip_len = 2 + (int)sqrtf((float)fft_len) + 1;
    int w_len = fft_len;
    hnd->int_work_area_len = ip_len;
    hnd->fp_work_area_len = w_len;
    hnd->int_work_area = (int*)HAL_MALLOC(ip_len * sizeof(int));
    HAL_SANITY_CHECK(NULL != hnd->int_work_area);
    memset(hnd->int_work_area, 0, ip_len * sizeof(int));
    hnd->heap_used += ip_len * sizeof(int);
    hnd->fp_work_area = (float*)HAL_MALLOC(w_len * sizeof(float));
    HAL_SANITY_CHECK(NULL != hnd->fp_work_area);
    memset(hnd->fp_work_area, 0, w_len * sizeof(float));
    hnd->heap_used += w_len * sizeof(float);

    return (_SwFftHnd_t)hnd;
}

static int _sw_fft(_SwFftHnd_t hnd, float *in, float *out, int fft_len, int isreal, int inverse, int sub_band)
{
    _SwFftMgr_t* _hnd = (_SwFftMgr_t*)hnd;
    AudioCallFunc_t* func = &_hnd->func;

    HAL_SANITY_CHECK(fft_len == _hnd->fft_len);

    if (isreal) {
        if (inverse) {
            float* tmp_buf = NULL;
            memcpy(out, in, sizeof(float) * fft_len);
            tmp_buf = out;
            Rdft(fft_len, -1, tmp_buf, _hnd->int_work_area, _hnd->fp_work_area);
            float temp = (sub_band == 0) ? (2.0f / (float)fft_len) : 2.0f;
            for (int j = 0; j < fft_len; j++) {
                tmp_buf[j] *= temp;
            }
        } else {
            float* tmp_buf = NULL;
            memcpy(out, in, sizeof(float) * fft_len);
            tmp_buf = out;
            Rdft(fft_len, 1, tmp_buf, _hnd->int_work_area, _hnd->fp_work_area);
        }
    } else {
        if (inverse) {
            memcpy(out, in, sizeof(float) * fft_len * 2);
            Cdft(2 * fft_len, 1, out, _hnd->int_work_area, _hnd->fp_work_area);
            float temp = (sub_band == 0) ? (1.0f / (float)fft_len) : 1.0f;
            for (int j = 0; j < fft_len * 2; j++) {
                out[j] *= 1.0f / ((float)fft_len);
            }
        } else {
            memcpy(out, in, sizeof(float) * fft_len * 2);
            Cdft(2 * fft_len, -1, out, _hnd->int_work_area, _hnd->fp_work_area);
        }
    }
    return 1;
}

static void _sw_fft_fini(_SwFftHnd_t hnd)
{
    _SwFftMgr_t* _hnd = (_SwFftMgr_t*)hnd;
    AudioCallFunc_t* func = &_hnd->func;

    HAL_SANITY_CHECK(NULL != _hnd);
    HAL_SANITY_CHECK(NULL != _hnd->int_work_area);
    HAL_SANITY_CHECK(NULL != _hnd->fp_work_area);
    HAL_FREE(_hnd->int_work_area);
    HAL_FREE(_hnd->fp_work_area);
    HAL_FREE(_hnd);
}

typedef struct {
    uint32_t cnt;
    uint32_t pre_max;
    uint32_t pre_avg;
    uint32_t pre_min;
    uint32_t model_max;
    uint32_t model_avg;
    uint32_t model_min;
    uint32_t post_max;
    uint32_t post_avg;
    uint32_t post_min;
    uint32_t mcu_max;
    uint32_t mcu_avg;
    uint32_t mcu_min;
    uint32_t npu_max;
    uint32_t npu_avg;
    uint32_t npu_min;
    uint32_t init_us;
} ModelPerf_t;

typedef enum {
    NPU_STATE_NULL = 0,
    NPU_STATE_INIT,
    NPU_STATE_IDLE,
    NPU_STATE_START,
    NPU_STATE_RUN,
    NPU_STATE_STOP,
    NPU_STATE_FINI,
} NpuState_t;
typedef struct {
    float fft_out[RFFT_LEN * 2];
    float block_in[BLOCK_LEN];
    float block_in_win[BLOCK_LEN];
    float ifft_out[RFFT_LEN * 2];
    float block_out[BLOCK_LEN];
    float mag[MAG_LEN];
    float phase[PHASE_LEN];
    float model_out[MODEL_OUT_CNT];

    TaskHandle_t xTaskHandle[1];
    _SwFftHnd_t fft_hnd;
    volatile NpuState_t npu_state;
    // SemaphoreHandle_t sema;

    ANSAICfg_t cfg;
    AudioCallFunc_t func;

    BRBHnd_t in_brb_hnd;
    BRBHnd_t out_brb_hnd;
    void* in_buf_ptr;
    void* out_buf_ptr;

    HciNpuHnd_t npu_hnd;
    ModelHnd_t model_hnd;
    // NpuCapability_t capability;
    // uint8_t npu_heap[NPU_HEAP_SIZE];
    // uint8_t npu_print_buf[NPU_PRINT_BUF_SIZE];

    /* MCU side */
    ModelPerf_t model_perf;

    ModleInfo_t info;
    int proc_en;
} DenoiseMgr_t;

// NPU_ATTRI_ALIGN8 PSRAM_DATA static float fft_out[RFFT_LEN * 2];
// NPU_ATTRI_ALIGN8 PSRAM_DATA static float block_in[BLOCK_LEN];
// NPU_ATTRI_ALIGN8 PSRAM_DATA static float block_in_win[BLOCK_LEN];
// NPU_ATTRI_ALIGN8 PSRAM_DATA static float ifft_out[RFFT_LEN * 2];
// NPU_ATTRI_ALIGN8 PSRAM_DATA static float block_out[BLOCK_LEN];

// NPU_ATTRI_ALIGN8 PSRAM_DATA static float mag[MAG_LEN];
// NPU_ATTRI_ALIGN8 PSRAM_DATA static float phase[PHASE_LEN];
// NPU_ATTRI_ALIGN8 PSRAM_DATA static float model_out[MODEL_OUT_CNT];

// extern volatile bool ANSTaskReady;

/* NPU side*/
NPU_ATTRI_ALIGN64 PSRAM_LOAD_DATA static i8 wgt[] = {
#include "./model/model.w"
};

/* model specific */
#include "./model/model.f"
NPU_ATTRI_ALIGN64 PSRAM_LOAD_DATA static float win_sqrt_han[] = {
#include "./model/window.f"
};

static void perf_init(ModelPerf_t* perf)
{
    perf->cnt = 0;
    perf->pre_max = 0;
    perf->model_max = 0;
    perf->post_max = 0;
    perf->mcu_max = 0;
    perf->npu_max = 0;
    perf->pre_avg = 0;
    perf->model_avg = 0;
    perf->post_avg = 0;
    perf->mcu_avg = 0;
    perf->npu_avg = 0;
    perf->pre_min = 0xFFFFFFFF;
    perf->model_min = 0xFFFFFFFF;
    perf->post_min = 0xFFFFFFFF;
    perf->mcu_min = 0xFFFFFFFF;
    perf->npu_min = 0xFFFFFFFF;
}

#define _ANSAI_ZX_VER_TO_STR(v)    #v
#define ANSAI_ZX_VER_TO_STR(v)     _ANSAI_ZX_VER_TO_STR(v)
static const char * _version = ANSAI_ZX_VER_TO_STR(LIBANSAI_ZX_VER);
static const char* ans_ai_get_version(void)
{
    return _version;
}

static void ans_ai_get_cfg(void *def_cfg)
{
    ANSAICfg_t *cfg = (ANSAICfg_t *)def_cfg;
    HAL_SANITY_CHECK(cfg != NULL);

    cfg->npu_log_level = LOG_LEVEL_WARNING;
    cfg->host_log_level = LOG_LEVEL_WARNING;
    cfg->perf_frame_interval = 100;
    cfg->task_priority = 4;
    cfg->cpu_pll = 600000000;
    cfg->npu_pll = 240000000;
}

// static void UserModelVerify(ModelVerifyArg_t* arg)
// {
//     // TEST_PRINTF("model verify, seq=%d, layer=%d, o_idx=%d, addr=0x%x, size=%d\n", 
//     //     arg->seq, arg->layer_idx, arg->result_idx, arg->result, arg->result_len);
//     // verify_layer(arg->seq, arg->layer_idx, arg->result_idx, arg->result_type, 
//     //     arg->result, arg->result_len);
// }

#define VERSION(major, minor, patch, reserved) \
    (((uint32_t)major << 24) + ((uint32_t)minor << 16) + ((uint32_t)patch << 8) + ((uint32_t)reserved))

static uint8_t ver_major_min = 2;
static uint8_t ver_minor_min = 0;
static uint8_t ver_patch_min = 2;
static uint8_t ver_reserved_min = 0;
static uint8_t ver_major_max = 2;
static uint8_t ver_minor_max = 0;
static uint8_t ver_patch_max = 2;
static uint8_t ver_reserved_max = 0;

static int _version_check(NpuVersion_t* npu_ver)
{
    uint32_t version = VERSION(npu_ver->major, npu_ver->minor, npu_ver->patch, 0);
    uint32_t version_min = VERSION(ver_major_min, ver_minor_min, ver_patch_min, ver_reserved_min);
    uint32_t version_max = VERSION(ver_major_max, ver_minor_max, ver_patch_max, ver_reserved_max);
    if (version > version_max) {
        return 1;
    } else if (version < version_min) {
        return -1;
    } else {
        return 0;
    }
}

static void _model_init(DenoiseMgr_t *denoise_mgr)
{
    HciVersion_t hci_ver;
    NpuVersion_t npu_ver;
    // HciNpuConf_t conf = HCI_NPU_DEFAULT_CONF();

    HAL_SANITY_CHECK(denoise_mgr != NULL);
    denoise_mgr->npu_hnd = HciNpuGetHnd(NPU_IP0);
    HAL_SANITY_CHECK(denoise_mgr->npu_hnd != NULL);

    // SetHostLogLevel(denoise_mgr->cfg.host_log_level);
    // denoise_mgr->npu_hnd = HciNpuInit(NPU_IP0, denoise_mgr->cfg.task_priority);

    // conf.heap_addr = (void*)denoise_mgr->npu_heap;
    // conf.heap_size = NPU_HEAP_SIZE;
    // conf.fast_math = 1;
    // HciNpuConf(denoise_mgr->npu_hnd, &conf);

    // HciSetNpuLogLevel(denoise_mgr->npu_hnd, denoise_mgr->cfg.npu_log_level);
    HciGetVersion(denoise_mgr->npu_hnd, &hci_ver, &npu_ver);
    int ver_check = _version_check(&npu_ver);
    if (ver_check < 0) {
        ts_printf("npu v%d.%d.%d is too old\n", 
            npu_ver.major, npu_ver.minor, npu_ver.patch);
        vTaskDelay(100);
        HAL_SANITY_CHECK(ver_check == 0);
    } else if (ver_check > 0) {
        ts_printf("npu v%d.%d.%d is too new\n", 
            npu_ver.major, npu_ver.minor, npu_ver.patch);
        vTaskDelay(100);
        HAL_SANITY_CHECK(ver_check == 0);
    }

    // LOGI(denoise_mgr->cfg.host_log_level, "task0: hci version = %d.%d.%d, npu version = %d.%d.%d \n",
    //     hci_ver.major, hci_ver.minor, hci_ver.patch, npu_ver.major, npu_ver.minor, npu_ver.patch);
    // HciGetNpuCapability(denoise_mgr->npu_hnd, &denoise_mgr->capability);
    // //vTaskDelay(100);
    // /* enable model verify and register the callback function */
    // HciSetModelVerify(denoise_mgr->npu_hnd, 0, UserModelVerify);
    // /* initialization for model verify */
    // // model_verify_init(2);
    // /* verify weight, will be removed later */
    // // verify_layer(0, 0, 0, NPU_INT8, wgt1, sizeof(wgt1));

    denoise_mgr->info.head.model_info_size = 0;
    denoise_mgr->info.head.ver_major = 0;
    denoise_mgr->info.head.ver_minor = 0;
    denoise_mgr->info.head.ver_patch = 0;
    denoise_mgr->info.head.ver_reserved = 0;
    denoise_mgr->info.head.model_type = MODEL_TYPE_MODEL_GENERAL;
    denoise_mgr->info.head.layer_cnt = layers_cnt;
    denoise_mgr->info.head.weight_total_size = weight_total_size;
    denoise_mgr->info.head.in_total_size = MODEL_IN_CNT * 4;
    denoise_mgr->info.head.layer_seq_len = layer_seq_len;
    denoise_mgr->info.head.insert_list_len = insert_list_len;
    denoise_mgr->info.head.result_layers_len = result_layers_len;
    denoise_mgr->info.head.max_layer = layers_cnt;
    denoise_mgr->info.head.total_output_size = MODEL_OUT_CNT * 4;
    HAL_MEMCPY(denoise_mgr->info.layers, layers, sizeof(LayerInfo_t) * layers_cnt);
    HAL_MEMCPY(denoise_mgr->info.layer_seq, layer_seq, sizeof(i32) * layer_seq_len);
    denoise_mgr->model_hnd = HciModelInit(denoise_mgr->npu_hnd, &denoise_mgr->info, (void*)wgt, NULL);
    HAL_SANITY_CHECK(denoise_mgr->model_hnd != NULL);

    /* ??? Is it fine to use MEMSET to initialize float array ??? */
    HAL_MEMSET(denoise_mgr->block_in, 0, sizeof(denoise_mgr->block_in));
    // HAL_MEMSET(denoise_mgr->ifft_out, 0, sizeof(denoise_mgr->ifft_out));
    HAL_MEMSET(denoise_mgr->block_out, 0, sizeof(denoise_mgr->block_out));
}

#define PRE_POST_USE_CCL1   (0)
extern uint32_t L1_BIAS_Base;
extern void Dma1d(uint32_t tcdm_addr, uint32_t l2_addr, uint32_t length, int type);
extern float npu_fast_sqrtf(float x);

typedef struct {
    DenoiseMgr_t* denoise_mgr;
    float* in;
    // float* fft_out;
    // float* mag;
    // float* phase;
} ModelPreArg_t;
/* input a partial voice buffer, and output frame's F(R/I in value), Modulo, Phase(I/R in sin/cos) in frequency domain. */
// static void _model_pre(DenoiseMgr_t *denoise_mgr, float *in, float* fft_out, float* mag, float* phase)
static void _model_pre(void* arg)
{
    DenoiseMgr_t *denoise_mgr = ((ModelPreArg_t*)arg)->denoise_mgr;
    float *in = ((ModelPreArg_t*)arg)->in;
    float* fft_out = denoise_mgr->fft_out;
    float* mag = denoise_mgr->mag;
    float* phase = denoise_mgr->phase;

    float *block_in = denoise_mgr->block_in;
    float *block_in_win = denoise_mgr->block_in_win;

    uint32_t rfft_size = RFFT_LEN * 2 * sizeof(float);
    uint32_t mag_size = MAG_LEN * sizeof(float);
    uint32_t phase_size = PHASE_LEN * sizeof(float);
    uint32_t shift_size = BLOCK_SHIFT * sizeof(float);
    uint32_t block_size = BLOCK_LEN * sizeof(float);
    uint32_t overlap_size = block_size - shift_size;

#if PRE_POST_USE_CCL1
    uint32_t l1_fft_out = CCL1_ALIGN_64(L1_BIAS_Base);
    uint32_t l1_block_in = CCL1_ALIGN_64(l1_fft_out + rfft_size);
    uint32_t l1_block_in_win = CCL1_ALIGN_64(l1_block_in + block_size);
    uint32_t l1_mag = CCL1_ALIGN_64(l1_block_in_win + block_size);
    uint32_t l1_phase = CCL1_ALIGN_64(l1_mag + mag_size);
    uint32_t l1_win_sqrt_han = CCL1_ALIGN_64(l1_phase + phase_size);

    Dma1d(l1_block_in, (uint32_t)block_in, block_size, 1);
    Dma1d(l1_win_sqrt_han, (uint32_t)win_sqrt_han, block_size, 1);

    // l1_block_in = (uint32_t)block_in;
    // l1_block_in_win = (uint32_t)block_in_win;
    // l1_mag = (uint32_t)mag;
    // l1_phase = (uint32_t)phase;
    // l1_win_sqrt_han = (uint32_t)win_sqrt_han;
#else
    uint32_t l1_fft_out = (uint32_t)fft_out;
    uint32_t l1_block_in = (uint32_t)block_in;
    uint32_t l1_block_in_win = (uint32_t)block_in_win;
    uint32_t l1_win_sqrt_han = (uint32_t)win_sqrt_han;
    uint32_t l1_mag = (uint32_t)mag;
    uint32_t l1_phase = (uint32_t)phase;
#endif
    /* move input into block_in buffer */
    memcpy((void*)l1_block_in, (void*)(l1_block_in + shift_size), overlap_size);
    memcpy((void*)(l1_block_in + overlap_size), &in[0], shift_size);
    /* add window */
    for (int i = 0; i < BLOCK_LEN; i++) {
        ((float*)l1_block_in_win)[i] = ((float*)l1_block_in)[i] * ((float*)l1_win_sqrt_han)[i];
    }
    /* rfft */
    _sw_fft(denoise_mgr->fft_hnd, (float*)l1_block_in_win, (float*)l1_fft_out, BLOCK_LEN, 1, 0, 0);

    /* get magnitude and phase from the result of rfft */
    float temp = ((float*)l1_fft_out)[0];
    ((float*)l1_mag)[0] = temp >= 0.0f ? temp : -temp;
    ((float*)l1_phase)[0] = 1.0f;
    ((float*)l1_phase)[1] = 0.0f;
    for (int32_t i = 1; i < RFFT_LEN - 1; i++) {
        float real = ((float*)l1_fft_out)[2 * i];
        float image = -((float*)l1_fft_out)[2 * i + 1];

        ((float*)l1_mag)[i] = npu_fast_sqrtf(real * real + image * image);
        if (((float*)l1_mag)[i] != 0.0f) {
            ((float*)l1_phase)[2 * i] = real / ((float*)l1_mag)[i];
            ((float*)l1_phase)[2 * i + 1] = image / ((float*)l1_mag)[i];
        } else {
            ((float*)l1_phase)[2 * i] = 1.0f;
            ((float*)l1_phase)[2 * i + 1] = 0.0f;
        }
    }
    temp = ((float*)l1_fft_out)[1];
    ((float*)l1_mag)[RFFT_LEN - 1] = temp >= 0.0f ? temp : -temp;
    ((float*)l1_phase)[PHASE_LEN - 2] = 1.0f;
    ((float*)l1_phase)[PHASE_LEN - 1] = 0.0f;
#if PRE_POST_USE_CCL1
    Dma1d(l1_block_in, (uint32_t)block_in, block_size, 0);
    Dma1d(l1_mag, (uint32_t)mag, mag_size, 0);
    Dma1d(l1_phase, (uint32_t)phase, phase_size, 0);
#endif
}

typedef struct {
    DenoiseMgr_t* denoise_mgr;
    // float* phase;
    // float* model_out;
    float* out;
} ModelPostArg_t;
/* adjust input buffer in frequency domain's P(I/R) with model1's output factor and then IFFT to time domain. */
/* ??? why don't apply the frequency domain's Modulo back ??? */
// static void _model_post(DenoiseMgr_t *denoise_mgr, float *phase, float* model_out, float *out)
static void _model_post(void* arg)
{
    DenoiseMgr_t *denoise_mgr = ((ModelPostArg_t*)arg)->denoise_mgr;
    float* phase = denoise_mgr->phase;
    float* model_out = denoise_mgr->model_out;
    float* out = ((ModelPostArg_t*)arg)->out;

    float *ifft_out = denoise_mgr->ifft_out;
    float *block_out = denoise_mgr->block_out;

    uint32_t rfft_size = RFFT_LEN * 2 * sizeof(float);
    uint32_t mag_size = MAG_LEN * sizeof(float);
    uint32_t phase_size = PHASE_LEN * sizeof(float);
    uint32_t shift_size = BLOCK_SHIFT * sizeof(float);
    uint32_t block_size = BLOCK_LEN * sizeof(float);
    uint32_t overlap_size = block_size - shift_size;
#if PRE_POST_USE_CCL1
    uint32_t l1_ifft_out = CCL1_ALIGN_64(L1_BIAS_Base);
    uint32_t l1_block_out = CCL1_ALIGN_64(l1_ifft_out + rfft_size);
    uint32_t l1_phase = CCL1_ALIGN_64(l1_block_out + block_size);
    uint32_t l1_model_out = CCL1_ALIGN_64(l1_phase + phase_size);
    uint32_t l1_out = CCL1_ALIGN_64(l1_model_out + mag_size);

    Dma1d(l1_block_out, (uint32_t)block_out, block_size, 1);
    Dma1d(l1_phase, (uint32_t)phase, phase_size, 1);
    Dma1d(l1_model_out, (uint32_t)model_out, mag_size, 1);
#else
    uint32_t l1_ifft_out = (uint32_t)ifft_out;
    uint32_t l1_block_out = (uint32_t)block_out;
    uint32_t l1_phase = (uint32_t)phase;
    uint32_t l1_model_out = (uint32_t)model_out;
    uint32_t l1_out = (uint32_t)out;
#endif
    /* update the input of irfft */
    for (int i = 0; i < RFFT_LEN; i++) {
        ((float*)l1_phase)[2 * i] *= ((float*)l1_model_out)[i];
        ((float*)l1_phase)[2 * i + 1] *= -((float*)l1_model_out)[i];
    }
    ((float*)l1_phase)[1] = ((float*)l1_phase)[2 * RFFT_LEN - 2];
    /* do irfft */
    _sw_fft(denoise_mgr->fft_hnd, ((float*)l1_phase), (float*)l1_ifft_out, BLOCK_LEN, 1, 1, 0);
    // /* move processed voice into voice_out_buf */
    // memcpy((void*)l1_block_out, (void*)(l1_block_out + shift_size), overlap_size);
    // memset((void*)(l1_block_out + overlap_size), 0, shift_size);
    // for (int32_t i = 0; i < BLOCK_LEN; i++) {
    //     ((float*)l1_block_out)[i] += ((float*)l1_ifft_out)[i];
    // }
    // for (int32_t i = 0; i < BLOCK_SHIFT; i++) {
    //     ((float*)l1_out)[i] = ((float*)l1_block_out)[i];// * 2.2276f;
    // }
    // add overlap part
    for (int i = 0; i < BLOCK_OVERLAP; i++) {
        ((float*)l1_block_out)[i] += ((float*)l1_ifft_out)[i];
    }
    // copy end part
    HAL_MEMCPY((void*)(l1_block_out + overlap_size), (void*)(l1_ifft_out + overlap_size), shift_size);
    /* move result out and shift block_out */
    for (int32_t i = 0; i < BLOCK_SHIFT; i++) {
        ((float*)l1_out)[i] = ((float*)l1_block_out)[i];
    }
    HAL_MEMCPY((void*)(l1_block_out), (void*)(l1_block_out + shift_size), overlap_size);
#if PRE_POST_USE_CCL1
    Dma1d(l1_block_out, (uint32_t)block_out, block_size, 0);
    Dma1d(l1_out, (uint32_t)out, shift_size, 0);
#endif
}

static int ans_ai_process(void *hnd, void *param)
{
    HAL_SANITY_CHECK(NULL != hnd && NULL != param);
    DenoiseMgr_t *denoise_mgr = (DenoiseMgr_t *)hnd;
    BRBHnd_t in_brb_hnd = denoise_mgr->in_brb_hnd;
    BRBHnd_t out_brb_hnd = denoise_mgr->out_brb_hnd;
    float *in_buf = NULL;
    float* out_buf = NULL;
    ANSAI_proc_param *proc_param = (ANSAI_proc_param *)param;
    float *in = proc_param->in;
    int in_len = proc_param->in_len;
    float *out = proc_param->out;
    int *out_len = &proc_param->out_len;
    HAL_SANITY_CHECK(NULL != in && NULL != out);
    HAL_SANITY_CHECK(BLOCK_SHIFT == in_len);

    if (denoise_mgr->npu_state != NPU_STATE_RUN) {
        for(int32_t i = 0; i < BLOCK_SHIFT; i++) {
            out[i] = in[i];
        }
        *out_len = BLOCK_SHIFT;
        LOGW(denoise_mgr->cfg.host_log_level, "ANS-AI state: %d\n", denoise_mgr->npu_state);
        return 0;
    }

    /* Dynamic bypass data */
    if (!denoise_mgr->proc_en) {
        for(int32_t i = 0; i < BLOCK_SHIFT; i++) {
            out[i] = in[i];
        }
        *out_len = BLOCK_SHIFT;
        if (HAL_SEMA_TAKE(out_brb_hnd->sem, 0) == pdTRUE) {
            BRB_Pop_RdPtr(out_brb_hnd);
        }
        return 0;
    }

    /* put input data into in_brb */
    do {
        in_buf = (float*)BRB_Get_WtPtr(in_brb_hnd);
        if(in_buf == NULL) {
            vTaskDelay(1);
        }
    } while(in_buf == NULL);
    memcpy(in_buf, in, BLOCK_SHIFT * sizeof(float));
    BRB_Pop_WtPtr(in_brb_hnd);
    HAL_SEMA_GIVE(in_brb_hnd->sem);
    
    /* get output data from out_brb */
    static int bypass_cnt = 0;
    if (HAL_SEMA_TAKE(out_brb_hnd->sem, 0) == pdTRUE) {
        out_buf = (float*)BRB_Get_RdPtr(out_brb_hnd);
        BRB_Pop_RdPtr(out_brb_hnd);
        memcpy(out, out_buf, BLOCK_SHIFT * sizeof(float));
        
        denoise_mgr->model_perf.npu_max = denoise_mgr->model_perf.model_max +
                denoise_mgr->model_perf.pre_max + denoise_mgr->model_perf.post_max;
        denoise_mgr->model_perf.npu_avg = denoise_mgr->model_perf.model_avg +
                denoise_mgr->model_perf.pre_avg + denoise_mgr->model_perf.post_avg;
        denoise_mgr->model_perf.npu_min = denoise_mgr->model_perf.model_min +
                denoise_mgr->model_perf.pre_min + denoise_mgr->model_perf.post_min;
        denoise_mgr->model_perf.cnt++;
        if (denoise_mgr->model_perf.cnt >= denoise_mgr->cfg.perf_frame_interval) {
            denoise_mgr->model_perf.pre_avg /= denoise_mgr->model_perf.cnt;
            denoise_mgr->model_perf.model_avg /= denoise_mgr->model_perf.cnt;
            denoise_mgr->model_perf.post_avg /= denoise_mgr->model_perf.cnt;
            denoise_mgr->model_perf.npu_avg /= denoise_mgr->model_perf.cnt;
            TEST_PRINTF("npu ansai_zx max/avg/min: %d/%d/%d\n",
                    denoise_mgr->model_perf.npu_max, denoise_mgr->model_perf.npu_avg, denoise_mgr->model_perf.npu_min);
            perf_init(&denoise_mgr->model_perf);
        }
    } else {
        bypass_cnt++;
        *out_len = 0;
        LOGW(denoise_mgr->cfg.host_log_level,
                "%s _bypass: %d\n", __func__, bypass_cnt);
        return -1;
    }
    *out_len = BLOCK_SHIFT;

    return 0;
}

static void _model_task(void *param)
{
    DenoiseMgr_t *denoise_mgr = (DenoiseMgr_t *)param;
    HAL_SANITY_CHECK(NULL != denoise_mgr);

    float *in = NULL, *out = NULL;
    ModelProcRsp_t rsp;
    BRBHnd_t in_brb_hnd = denoise_mgr->in_brb_hnd;
    BRBHnd_t out_brb_hnd = denoise_mgr->out_brb_hnd;

    _model_init(denoise_mgr);
    PERF_END(init);
    denoise_mgr->model_perf.init_us = PERF_DATA(init) / (denoise_mgr->cfg.cpu_pll / 2000000);

    while (1) {
        switch (denoise_mgr->npu_state)
        {
        case NPU_STATE_RUN:
            if (HAL_SEMA_TAKE(in_brb_hnd->sem, 10) == pdTRUE) {
                in = (float*)BRB_Get_RdPtr(in_brb_hnd);
                if(in == NULL)
                    continue;

                do {
                    out = (float*)BRB_Get_WtPtr(out_brb_hnd);
                    if(out == NULL) {
                        vTaskDelay(1);
                    }
                } while(out == NULL);

                PERF_INIT(model);
                PERF_INIT(pre);
                PERF_INIT(post);
                // HAL_LOCK(denoise_mgr->sema);
                if (denoise_mgr->proc_en) {
                    /* pre-process */
                    PERF_START(pre);
                    ModelPreArg_t pre_arg;
                    pre_arg.denoise_mgr = denoise_mgr;
                    pre_arg.in = in;
                    // pre_arg.fft_out = denoise_mgr->fft_out;
                    // pre_arg.mag = denoise_mgr->mag;
                    // pre_arg.phase = denoise_mgr->phase;
                    HciExecuteFunc(denoise_mgr->npu_hnd, _model_pre, &pre_arg, NULL);
                    PERF_END(pre);
                    if (denoise_mgr->model_perf.cnt == 1) {
                        _print_v_f("phase:", denoise_mgr->phase, 0, 16);
                        _print_v_f("mag:", denoise_mgr->mag, 0, 16);
                    }
                    /* process */
                    PERF_START(model);
                    HciModelProc(denoise_mgr->npu_hnd, denoise_mgr->model_hnd, 
                        (void*)(denoise_mgr->mag), (void*)(denoise_mgr->model_out), &rsp);
                    PERF_END(model);
                    if (denoise_mgr->model_perf.cnt == 1) {
                        _print_v_f("model:", denoise_mgr->model_out, 0, 16);
                    }
                    /* post-process */
                    PERF_START(post);
                    ModelPostArg_t post_arg;
                    post_arg.denoise_mgr = denoise_mgr;
                    // post_arg.phase = denoise_mgr->phase;
                    // post_arg.model_out = denoise_mgr->model_out;
                    post_arg.out = out;
                    HciExecuteFunc(denoise_mgr->npu_hnd, _model_post, &post_arg, NULL);
                    PERF_END(post);
                    if (denoise_mgr->model_perf.cnt == 1) {
                        _print_v_f("out:", out, 0, 16);
                    }
                } else {
                    PERF_START(pre);
                    PERF_END(pre);
                    PERF_START(model);
                    rsp.time_us = 0;
                    HAL_MEMCPY(out, in, BLOCK_SHIFT * sizeof(float));
                    PERF_END(model);
                    PERF_START(post);
                    PERF_END(post);
                }
                // HAL_UNLOCK(denoise_mgr->sema);
                uint32_t pre_us = PERF_DATA(pre) / (denoise_mgr->cfg.cpu_pll / 2000000);
                if (pre_us > denoise_mgr->model_perf.pre_max) 
                    denoise_mgr->model_perf.pre_max = pre_us;
                if (pre_us < denoise_mgr->model_perf.pre_min) 
                    denoise_mgr->model_perf.pre_min = pre_us;
                denoise_mgr->model_perf.pre_avg += pre_us;
                uint32_t model_us = PERF_DATA(model) / (denoise_mgr->cfg.cpu_pll / 2000000);
                if (model_us > denoise_mgr->model_perf.model_max)
                    denoise_mgr->model_perf.model_max = model_us;
                if (model_us < denoise_mgr->model_perf.model_min)
                    denoise_mgr->model_perf.model_min = model_us;
                denoise_mgr->model_perf.model_avg += model_us;
                uint32_t post_us = PERF_DATA(post) / (denoise_mgr->cfg.cpu_pll / 2000000);
                if (post_us > denoise_mgr->model_perf.post_max)
                    denoise_mgr->model_perf.post_max = post_us;
                if (post_us < denoise_mgr->model_perf.post_min)
                    denoise_mgr->model_perf.post_min = post_us;
                denoise_mgr->model_perf.post_avg += post_us;

                BRB_Pop_RdPtr(in_brb_hnd);
                BRB_Pop_WtPtr(out_brb_hnd);
                HAL_SEMA_GIVE(out_brb_hnd->sem);
            }
            break;
        case NPU_STATE_IDLE:
            vTaskDelay(1);
            break;
        case NPU_STATE_INIT:
            denoise_mgr->npu_state = NPU_STATE_IDLE;
            break;
        case NPU_STATE_START:
            denoise_mgr->npu_state = NPU_STATE_RUN;
            break;
        case NPU_STATE_STOP:
            denoise_mgr->npu_state = NPU_STATE_IDLE;
            break;
        case NPU_STATE_FINI:
            goto npu_destroy;
        default:
            LOGE(denoise_mgr->cfg.host_log_level, "%s, npu state error\n", __func__);
            break;
        }
    }

npu_destroy:
    if (denoise_mgr->model_hnd)
        HciModelFini(denoise_mgr->npu_hnd, denoise_mgr->model_hnd);
    // if (denoise_mgr->npu_hnd)
    //     HciNpuFini(denoise_mgr->npu_hnd);
    denoise_mgr->npu_state = NPU_STATE_NULL;
    vTaskDelete(NULL);
}

static void _create_task(DenoiseMgr_t *denoise_mgr)
{
    BaseType_t   xTask;

	xTask = xTaskCreate(_model_task, "_model_task",
			configMINIMAL_STACK_SIZE * 2, (void *)denoise_mgr,
			denoise_mgr->cfg.task_priority, &denoise_mgr->xTaskHandle[0]);
    HAL_SANITY_CHECK(xTask == pdPASS); 
}

static void *ans_ai_init(void *param)
{
    ANSAICfg_t *cfg;
    DenoiseMgr_t *denoise_mgr;
    AudioCallFunc_t* func;

    PERF_START(init);
    if (!param)
        return NULL;

    cfg = ((ANSAI_init_param *)param)->conf;
    func = ((ANSAI_init_param *)param)->func;
    if (!cfg || !func)
        return NULL;

    denoise_mgr = (DenoiseMgr_t *)PsramMalloc(sizeof(DenoiseMgr_t));
    if (!denoise_mgr) {
        LOGE(LOG_LEVEL_ERROR, "%s malloc denoise_mgr fail\n", __func__);
        return NULL;
    }
    memset(denoise_mgr, 0, sizeof(DenoiseMgr_t));
    memcpy(&denoise_mgr->cfg, cfg, sizeof(ANSAICfg_t));
    memcpy(&denoise_mgr->func, func, sizeof(AudioCallFunc_t));
    denoise_mgr->npu_state = NPU_STATE_NULL;

    perf_init(&denoise_mgr->model_perf);

    int frame_size = BLOCK_SHIFT * sizeof(float);
    denoise_mgr->in_buf_ptr = PsramMalloc(frame_size * MODEL_BUF_CNT);
    HAL_SANITY_CHECK(NULL != denoise_mgr->in_buf_ptr);
    HAL_MEMSET(denoise_mgr->in_buf_ptr, 0, frame_size * MODEL_BUF_CNT);
    denoise_mgr->in_brb_hnd = BRB_Init(MODEL_BUF_CNT, frame_size,
            denoise_mgr->in_buf_ptr);
    HAL_SANITY_CHECK(NULL != denoise_mgr->in_brb_hnd);

    frame_size = BLOCK_SHIFT * sizeof(float);
    denoise_mgr->out_buf_ptr = PsramMalloc(frame_size * MODEL_BUF_CNT);
    HAL_SANITY_CHECK(NULL != denoise_mgr->out_buf_ptr);
    denoise_mgr->out_brb_hnd = BRB_Init(MODEL_BUF_CNT, frame_size,
            denoise_mgr->out_buf_ptr);
    HAL_SANITY_CHECK(NULL != denoise_mgr->out_brb_hnd);

    denoise_mgr->fft_hnd = _sw_fft_init(BLOCK_LEN, &denoise_mgr->func);

    denoise_mgr->npu_state = NPU_STATE_INIT;
    _create_task(denoise_mgr);
    denoise_mgr->proc_en = cfg->proc_en;

    return (void *)denoise_mgr;
}

static void ans_ai_fini(void *hnd)
{
    DenoiseMgr_t *denoise_mgr = (DenoiseMgr_t *)hnd;

    HAL_SANITY_CHECK(NULL != denoise_mgr);

    HAL_SANITY_CHECK(denoise_mgr->npu_state != NPU_STATE_NULL);
    while (denoise_mgr->npu_state != NPU_STATE_IDLE) {
        vTaskDelay(1);
    }
    denoise_mgr->npu_state = NPU_STATE_FINI;
    while (denoise_mgr->npu_state != NPU_STATE_NULL) {
        vTaskDelay(1);
    }

    BRB_Fini(denoise_mgr->in_brb_hnd);
    PsramFree(denoise_mgr->in_buf_ptr);
    BRB_Fini(denoise_mgr->out_brb_hnd);
    PsramFree(denoise_mgr->out_buf_ptr);

    _sw_fft_fini(denoise_mgr->fft_hnd);

    PsramFree(denoise_mgr);
}

static int ans_ai_start(void *hnd, void *param)
{
    DenoiseMgr_t *denoise_mgr = (DenoiseMgr_t *)hnd;
    HAL_SANITY_CHECK(NULL != denoise_mgr);
    HAL_SANITY_CHECK(denoise_mgr->npu_state != NPU_STATE_NULL);
    while (denoise_mgr->npu_state != NPU_STATE_IDLE) {
        vTaskDelay(1);
    }
    denoise_mgr->npu_state = NPU_STATE_RUN;
    return 0;
}

static int ans_ai_stop(void *hnd, void *param)
{
    DenoiseMgr_t *denoise_mgr = (DenoiseMgr_t *)hnd;
    HAL_SANITY_CHECK(NULL != denoise_mgr);
    HAL_SANITY_CHECK(denoise_mgr->npu_state != NPU_STATE_NULL);
    if (denoise_mgr->npu_state == NPU_STATE_IDLE) {
        LOGE(denoise_mgr->cfg.host_log_level, "%s, already stoped\n", __func__);
        return -1;
    }
    while (denoise_mgr->npu_state != NPU_STATE_RUN) {
        vTaskDelay(1);
    }
    denoise_mgr->npu_state = NPU_STATE_STOP;
    /* wait stop finish */
    while (denoise_mgr->npu_state != NPU_STATE_IDLE) {
        vTaskDelay(1);
    }
    return 0;
}

static int ans_ai_ctrl(void *hnd, int cmd, void *param)
{
    DenoiseMgr_t *denoise_mgr;
    HAL_SANITY_CHECK(NULL != hnd && param != NULL);
    denoise_mgr = (DenoiseMgr_t *)hnd;

    if (cmd == ANSAI_CMD_PROC_EN) {
        ANSAI_ctrl_param *ctrl_param = (ANSAI_ctrl_param *)param;
        denoise_mgr->proc_en = ctrl_param->proc_en;
    }

    return 0;
}

static struct alg_ops _alg_ops = {
    .get_version = ans_ai_get_version,
    .get_cfg = ans_ai_get_cfg,
    .init  = ans_ai_init,
    .proc  = ans_ai_process,
    .ctrl  = ans_ai_ctrl,
    .start = ans_ai_start,
    .stop  = ans_ai_stop,
    .fini  = ans_ai_fini,
};

ALG_MODULE(ALG_TYPE_ANSAI, _alg_ops);
