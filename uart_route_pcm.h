/**
 * @file uart_route_pcm.h
 * @brief   Pcm function function interface.
 *          This file provides concise but complete functions to manage the Pcm.
 *          This function is based on uart_route.
 * @author Zengxin (Zengxin@timesintelli.com)
 * @version 1.0
 * @date 2023-12-12
 * @copyright Copyright (c) 2023 Timesintelli, Inc
 */
#ifndef _UART_ROUTE_PCM_H
#define _UART_ROUTE_PCM_H

#include "ts_sys.h"
#include "cmd.h"
#include "ts_hal_device.h"

#define ROUTEPCM_MSGQUEUE_SIZE  5
#define ROUTEPCM_MAX_BUF_SIZE   (64 * 1024)
#define ROUTEPCM_MAX_VALUE_SIZE (16 * 1024)
/**
 * @brief Pcm command defination
 */
typedef enum _PCM_SUBCMD_T {
    PCM_SUBCMD_START = 0x0,
    PCM_SUBCMD_STREAM,
    PCM_SUBCMD_END,
    PCM_SUBCMD_RESPONSE,
} PCM_SUBCMD_T;
/**
 * @brief Pcm command status defination
 */
typedef enum _PCM_FLAG_T {
    PCM_FLAG_REQUEST = 0x0,
    PCM_FLAG_RESPONSE_RESULT,
    PCM_FLAG_RESPONSE_ACK,
    PCM_FLAG_RESPONSE_ERR,
} PCM_FLAG_T;
/**
 * @brief Pcm message structure defination
 */
typedef struct {
    uint8_t  src_id;  /**ID of the source*/
    uint8_t *payload; /**The payload of Pcm*/
} __attribute__((packed)) pcm_msg_t;
/**
 * @brief RoutePcm command structure defination
 */
typedef struct {
    uint8_t sub_cmd; /**Pcm command*/
    uint8_t flag;  /**0: Request; 1: Respose ack, 2: Respose err*/
    uint8_t idx;  /**Pcm idx*/
    uint16_t size;    /**content size*/
    int16_t *content;   /**content pointer*/
} __attribute__((packed)) pcm_cmd_t;

/**
 * @brief Default routePcm configuration
 * Note:The user must define chnhnd and norhnd.
 */
#define ROUTEPCM_DEFAULT_CONFIG() \
    { \
        .chnhnd         = NULL, \
        .norhnd         = NULL, \
        .max_value_size = ROUTEPCM_MAX_VALUE_SIZE, \
        .max_buf_size   = ROUTEPCM_MAX_BUF_SIZE, \
    }
/**
 * @brief RoutePcm configuration structure defination
 */
typedef struct RoutePcmConf {
    // uint32_t      max_value_size;
    // uint32_t      max_buf_size;
    RouteChnHnd_t chnhnd;
} RoutePcmConf_t;
/**
 * @brief RoutePcm handler structure defination
 */
typedef struct RoutePcmMgr {
    HAL_LOCK_T      lock;
    RouteChnHnd_t   chnhnd;
    // uint32_t        max_value_size;
    // uint32_t        max_buf_size;
    uint8_t         old_idx;
    // TaskHandle_t    taskhnd;
    HAL_QUEU_T      rx_msg_hnd;
} RoutePcmMgr_t;
/**
 * @brief RoutePcm handler typedef
 */
typedef RoutePcmMgr_t *RoutePcmHnd_t;
/**
 * @brief Initialize RoutePcm.
 * @param[in] conf  Pointer to RoutePcm configure struct.
 * @return RoutePcmHnd_t Pointer to RoutePcm handler struct.
 */
RoutePcmHnd_t RoutePcm_Init(RoutePcmConf_t *conf);
/**
 * @brief De-initialize RoutePcm.
 * @param[in] hnd  Pointer to RoutePcm handler struct.
 */
void RoutePcm_Finit(RoutePcmHnd_t hnd);

#endif