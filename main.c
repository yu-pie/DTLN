#include <math.h>

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
#include "ts_psram.h"

#include "ppl.h"
#include "rtm_task.h"
// #include "pipeline.h"
#include "alg_module.h"
// #include "app.h"

#include "uart_route.h"
#include "uart_route_pcm.h"

#define SELF_CHIP_ID CHIP_ID_2
#define BAUD_RATE    4000000
#ifndef ROUTE_OTA_RX_BUF_SIZE
#define ROUTE_OTA_RX_BUF_SIZE (16 * 1024 + 128)
#endif
#define ROUTE_OTA_TX_BUF_SIZE (1024)

static RouteHnd_t    hnd         = NULL;
static RouteChnHnd_t chnhnd      = NULL;
static RoutePcmHnd_t RoutePcmHnd = NULL;

void uart_route_pcm_init(void)
{
    // uart_route init
    hnd                     = Route_Init(SELF_CHIP_ID);
    RouteChnConf_t chn_conf = {
        .UartNum            = 1,
        .BaudRate           = BAUD_RATE,
        .rx_buf_size        = ROUTE_OTA_RX_BUF_SIZE,
        .tx_buf_size        = ROUTE_OTA_TX_BUF_SIZE,
    };
    chnhnd = Route_OpenChannel(&chn_conf);
    HAL_SANITY_CHECK(chnhnd != NULL);

    // uart_route_ota init
    // RoutePcmConf_t pcm_conf = ROUTEOTA_DEFAULT_CONFIG();
    RoutePcmConf_t pcm_conf;
    pcm_conf.chnhnd = chnhnd;

    RoutePcmHnd = RoutePcm_Init(&pcm_conf);
    HAL_SANITY_CHECK(RoutePcmHnd != NULL);
}

void uart_route_pcm_fini(void)
{
    if (RoutePcmHnd != NULL) {
        RoutePcm_Finit(RoutePcmHnd);
        RoutePcmHnd = NULL;
    }
}

void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName)
{
    ts_tiny_printf("task: %s stack overflow!\n", pcTaskName);
}

#if (configSUPPORT_STATIC_ALLOCATION == 1)
static StaticTask_t IdleTaskTCB;
PSRAM_DATA static StackType_t IdleTaskStack[configMINIMAL_STACK_SIZE];

PSRAM_DATA static StackType_t TimerTaskStack[configTIMER_TASK_STACK_DEPTH];
static StaticTask_t TimerTaskTCB;

void vApplicationGetIdleTaskMemory(StaticTask_t **ppxIdleTaskTCBBuffer,
        StackType_t **ppxIdleTaskStackBuffer,
        uint32_t *pulIdleTaskStackSize)
{
    *ppxIdleTaskTCBBuffer=&IdleTaskTCB;
    *ppxIdleTaskStackBuffer=IdleTaskStack;
    *pulIdleTaskStackSize=configMINIMAL_STACK_SIZE;
}

void vApplicationGetTimerTaskMemory(StaticTask_t **ppxTimerTaskTCBBuffer,
        StackType_t **ppxTimerTaskStackBuffer,
        uint32_t *pulTimerTaskStackSize)
{
    *ppxTimerTaskTCBBuffer=&TimerTaskTCB;
    *ppxTimerTaskStackBuffer=TimerTaskStack;
    *pulTimerTaskStackSize=configTIMER_TASK_STACK_DEPTH;
}
#endif

int main(void)
{
    // sys_setdiv_uart(0x23);

    // UARTHnd_t uart_hnd = HAL_UART_Init(SYS_IP_UART2);
    // HAL_SANITY_CHECK(uart_hnd != NULL);
    // UARTConf_t uart_conf = UART_DEFAULT_CONFIG();
    // uart_conf.BaudRate   = 1000000;
    // HAL_UART_Config(uart_hnd, &uart_conf);

    ts_printf(">>>>>>>>>> %s <<<<<<<<<<<<<\n", getSDKVersion());
    enable_core_perf_cnt(0);

    uart_route_pcm_init();

    /* Start the scheduler. */
    vTaskStartScheduler();

    while (1)
        ;
    return 1;
}


