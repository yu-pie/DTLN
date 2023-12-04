#ifndef __LOG_DEF_H__
#define __LOG_DEF_H__

#include "ts_print.h"

enum {
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_INFO,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_MAX,
};

#define LOG_PRINTF ts_printf

#define LOGE(level, fmt, ...) \
    do { \
        if (level >= LOG_LEVEL_ERROR) { \
            LOG_PRINTF(fmt, __VA_ARGS__); \
        } \
    } while (0)

#define LOGW(level, fmt, ...) \
    do { \
        if (level >= LOG_LEVEL_WARNING) { \
            LOG_PRINTF(fmt, __VA_ARGS__); \
        } \
    } while (0)

#define LOGI(level, fmt, ...) \
    do { \
        if (level >= LOG_LEVEL_INFO) { \
            LOG_PRINTF(fmt, __VA_ARGS__); \
        } \
    } while (0)

#define LOGD(level, fmt, ...) \
    do { \
        if (level >= LOG_LEVEL_DEBUG) { \
            LOG_PRINTF(fmt, __VA_ARGS__); \
        } \
    } while (0)

#endif
