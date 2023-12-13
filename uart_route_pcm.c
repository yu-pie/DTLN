// #include <math.h>
#include <stdio.h>
/* FreeRTOS kernel includes. */
#include "FreeRTOS.h"
#include "semphr.h"
#include "task.h"
#include "priority_list.h"

/*AT1K includes*/
#include "ts_device.h"
#include "ts_hal_device.h"
#include "ts_print.h"
#include "ts_perf.h"

#include "uart_route.h"
#include "uart_route_pcm.h"

// #include "audio_def.h"

#define PCM_LOGE ts_tiny_printf
#define PCM_LOGD ts_tiny_printf
#define PCM_LOGI ts_tiny_printf

#define PCM_PROC_SOURCE_SIZE 256
#define PCM_PROC_RESULT_SIZE 256
int16_t  pcm_proc_source_buf[PCM_PROC_SOURCE_SIZE];
int16_t  pcm_proc_result_buf[PCM_PROC_RESULT_SIZE];

// extern float fast_sqrtf(float x);

static void DEBUG_PRINTF(uint8_t *buf, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++) {

        if (i % 32 == 0) {
            PCM_LOGD("\n");
        }
        PCM_LOGD("%02x ", buf[i]);
    }
    PCM_LOGD("\n");
}

static int pcm_callback(ROUTE_PACKET_t *packet, void *arg)
{
    uint8_t   *p;
    pcm_msg_t  datap      = {0};
    HAL_QUEU_T rx_msg_hnd = (HAL_QUEU_T)arg;

    PCM_LOGD("Receive a packet from id[0x%02d]: cmd %d, size %d\n", packet->src_id, packet->cmd, packet->size);

    datap.src_id  = packet->src_id;
    datap.payload = NULL;

    // DEBUG_PRINTF(packet->content, packet->size);
    if (packet->size) {
        p = (uint8_t *)HAL_MALLOC(packet->size);
        if (p) {
            HAL_MEMCPY(p, packet->content, packet->size);
            datap.payload = p;
            HAL_QUEU_PUT(rx_msg_hnd, &datap);
            PCM_LOGD("send msg\n");
        } else {
            PCM_LOGE("pcm_callback create buffer error\n");
        }
    }
    return 0;
}

static HAL_RET_T pcm_cmd_parse(uint8_t *buf_in, pcm_cmd_t *pcm_cmd)
{
    HAL_SANITY_CHECK(buf_in != NULL);
    HAL_SANITY_CHECK(pcm_cmd != NULL);
    pcm_cmd_t *cmd_in   = (pcm_cmd_t *)buf_in;
    pcm_cmd->sub_cmd    = cmd_in->sub_cmd;
    pcm_cmd->idx        = cmd_in->idx;
    pcm_cmd->size       = cmd_in->size;
    if (pcm_cmd->size > 0) {
        pcm_cmd->content = (int16_t *)(&buf_in[5]);
    }

    return HAL_OK;
}

static HAL_RET_T _pcm_cmd_response(RoutePcmHnd_t hnd, uint8_t dir_id, pcm_cmd_t *pcm_cmd)
{
    uint8_t *tx_buf = NULL;
    tx_buf          = HAL_MALLOC(5 + pcm_cmd->size);
    HAL_SANITY_CHECK(tx_buf != NULL);
    RouteChnHnd_t chnhnd    = hnd->chnhnd;
    pcm_cmd_t    *cmd_out   = (pcm_cmd_t *)tx_buf;
    cmd_out->sub_cmd        = pcm_cmd->sub_cmd;
    cmd_out->flag           = pcm_cmd->flag;
    cmd_out->idx            = pcm_cmd->idx;
    cmd_out->size           = pcm_cmd->size;
    if (cmd_out->size) {
        HAL_MEMCPY((uint8_t *)&cmd_out->content, pcm_cmd->content, pcm_cmd->size);
    }
    Route_WriteData(chnhnd, dir_id, PKT_CMD_PCM, (char *)tx_buf, pcm_cmd->size + 5, 10000);
    HAL_FREE(tx_buf);
    return HAL_OK;
}

static void pcm_cmd_response_ack(RoutePcmHnd_t hnd, uint8_t dir_id, uint8_t subcmd, uint8_t idx)
{
    pcm_cmd_t pcm_cmd   = {0};
    pcm_cmd.sub_cmd     = subcmd;
    pcm_cmd.flag        = PCM_FLAG_RESPONSE_ACK;
    pcm_cmd.idx         = idx;
    pcm_cmd.content     = NULL;
    pcm_cmd.size        = 0;
    _pcm_cmd_response(hnd, dir_id, &pcm_cmd);
}

static void pcm_cmd_response_err(RoutePcmHnd_t hnd, uint8_t dir_id, uint8_t subcmd, uint8_t idx)
{
    pcm_cmd_t pcm_cmd   = {0};
    pcm_cmd.sub_cmd     = subcmd;
    pcm_cmd.flag        = PCM_FLAG_RESPONSE_ERR;
    pcm_cmd.idx         = idx;
    pcm_cmd.content     = NULL;
    pcm_cmd.size        = 0;
    _pcm_cmd_response(hnd, dir_id, &pcm_cmd);
}

static void pcm_cmd_response(RoutePcmHnd_t hnd, uint8_t dir_id, uint8_t subcmd, uint8_t idx, int16_t *pcm_proc_result_buf, uint16_t pcm_proc_result_size)
{
    pcm_cmd_t pcm_cmd   = {0};
    pcm_cmd.sub_cmd     = subcmd;
    pcm_cmd.flag        = PCM_FLAG_RESPONSE_RESULT;
    pcm_cmd.idx         = idx;
    pcm_cmd.content     = pcm_proc_result_buf;
    pcm_cmd.size        = pcm_proc_result_size;
    _pcm_cmd_response(hnd, dir_id, &pcm_cmd);
}

static uint32_t get_frame_checksum(int16_t *content, uint16_t size)
{
    uint32_t frame_checksum = 0;
    for (uint16_t i = 0; i < size; i++) {
        frame_checksum += content[i];
    }
    return frame_checksum;
}

static void _pcm_task(void *arg)
{
    pcm_msg_t     rx_msg;
    pcm_cmd_t     pcm_cmd     = {0};
    RoutePcmHnd_t hnd         = (RoutePcmHnd_t)arg;
    HAL_QUEU_T    rx_msg_hnd  = hnd->rx_msg_hnd;
    while (1) {
        HAL_QUEU_GET(rx_msg_hnd, &rx_msg);
        PCM_LOGD("Receive a packet");
        HAL_LOCK(hnd->lock);
        // 解析PCM命令
        pcm_cmd_parse(rx_msg.payload, &pcm_cmd);

        switch (pcm_cmd.sub_cmd) {
        case PCM_SUBCMD_START: {
            PCM_LOGD("Start processing pcm.\n");
            pcm_cmd_response_ack(hnd, rx_msg.src_id, PCM_SUBCMD_RESPONSE, pcm_cmd.idx);
            break;
        }
        case PCM_SUBCMD_STREAM: { 
            if (pcm_cmd.idx == hnd->old_idx) {
                pcm_cmd_response(hnd, rx_msg.src_id, PCM_SUBCMD_RESPONSE, pcm_cmd.idx, pcm_proc_result_buf, PCM_PROC_RESULT_SIZE * 2);
            } else {
                // int32_t frame_checksum = get_frame_checksum(pcm_cmd.content, (pcm_cmd.size / 2));
                // TODO get the checksum
                // int32_t checksum = 0; 
                // if (frame_checksum != checksum) {
                //     PCM_LOGE("Checksum error!.Terminate.\n");
                //     pcm_cmd_response_err(hnd, rx_msg.src_id, PCM_SUBCMD_RESPONSE, pcm_cmd.idx);
                // } else {
                    hnd->old_idx = pcm_cmd.idx;
                    void* in;
                    in = pcm_cmd.content;
                    // TODO do processing
                    // ANSAI_proc_param proc_param;
                    // proc_param.in = pcm_proc_source_buf;
                    // proc_param.in_len = shift_len;
                    // proc_param.out = pcm_proc_result_buf;
                    // proc_param.out_len = 0;
                    // ALG_PROC(alg_ops, alg_hnd, &proc_param);

                    pcm_cmd_response(hnd, rx_msg.src_id, PCM_SUBCMD_RESPONSE, pcm_cmd.idx, pcm_proc_result_buf, PCM_PROC_RESULT_SIZE * 2);
                // }
            }
            break;
        }
        case PCM_SUBCMD_END: {
            PCM_LOGD("End processing pcm.\n");
            pcm_cmd_response_ack(hnd, rx_msg.src_id, PCM_SUBCMD_RESPONSE, pcm_cmd.idx);
            break;
        }
        default:
            break;
        }
        if (rx_msg.payload) {
            HAL_FREE(rx_msg.payload);
        }
        HAL_UNLOCK(hnd->lock);
    }
}

RoutePcmHnd_t RoutePcm_Init(RoutePcmConf_t *conf)
{
    HAL_SANITY_CHECK(conf != NULL);
    HAL_SANITY_CHECK(conf->chnhnd != NULL);
    RoutePcmHnd_t hnd = HAL_MALLOC(sizeof(RoutePcmMgr_t));
    HAL_SANITY_CHECK(hnd != NULL);
    hnd->chnhnd     = conf->chnhnd;
    hnd->rx_msg_hnd = HAL_QUEU_INIT(ROUTEPCM_MSGQUEUE_SIZE, sizeof(pcm_msg_t));
    HAL_SANITY_CHECK(NULL != hnd->rx_msg_hnd);
    Route_RegisterCmdCallback(PKT_CMD_PCM, pcm_callback, hnd->rx_msg_hnd);
    // HAL_SANITY_CHECK(conf->max_buf_size != 0);
    // HAL_SANITY_CHECK(conf->max_value_size < hnd->chnhnd->rx_buf_size);
    // hnd->max_buf_size   = conf->max_buf_size;
    // hnd->max_value_size = conf->max_value_size;
    BaseType_t xTask;
    // xTask = xTaskCreate(_pcm_task, "_pcm_task",
    //                     configMINIMAL_STACK_SIZE * 2, (void *)hnd,
    //                     tskIDLE_PRIORITY + 2, &hnd->taskhnd);
    xTask = xTaskCreate(_pcm_task, "_pcm_task",
                    configMINIMAL_STACK_SIZE * 2, (void *)hnd,
                    tskIDLE_PRIORITY + 2, NULL);
    HAL_SANITY_CHECK(xTask == pdPASS);
    HAL_LOCK_INIT(hnd->lock);
    return hnd;
}

void RoutePcm_Finit(RoutePcmHnd_t hnd)
{
    HAL_SANITY_CHECK(hnd->lock != NULL);
    HAL_LOCK(hnd->lock);
    Route_UnregisterCmdCallback(PKT_CMD_PCM);
    // vTaskDelete(hnd->taskhnd);
    HAL_QUEU_FINI(hnd->rx_msg_hnd);
    HAL_UNLOCK(hnd->lock);
    HAL_LOCK_FINI(hnd->lock);
    HAL_FREE(hnd);
}
