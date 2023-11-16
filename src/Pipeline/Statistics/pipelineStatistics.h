#ifndef _PIPELIESTATISTICS_H
#define _PIPELIESTATISTICS_H

typedef unsigned long int ulong;

namespace statistics{
    struct pipelineStatistics{
        ulong loopLength;
        double averageLoopLength;
        ulong maxLoopLength;
        double averageLoopLatency;
        double averageWorkTime;
        ulong maxWorkTime;
        double averageLoad;
        double maxLoad;
        double userInputLatency;
    };
}

#endif
