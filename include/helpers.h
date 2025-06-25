// helpers.h
#pragma once
#include <stdlib.h>

static inline const char *getenv_or(const char *name, const char *fallback)
{
    const char *v = getenv(name);
    return (v && *v) ? v : fallback;
}
