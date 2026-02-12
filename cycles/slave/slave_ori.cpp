#include <crts.h>
#include <slave.h>
#include "slave_math.h"

#include "kernel/kernel_compat_cpu.h"
#include "kernel/kernel_math.h"
#include "kernel/kernel_types.h"
#include "kernel/split/kernel_split_data.h"
#include "kernel/kernel_globals.h"
#include "kernel/kernel_color.h"
#include "kernel/kernels/cpu/kernel_cpu_image.h"
#include "kernel/kernel_film.h"
#include "kernel/kernel_path.h"
#include "kernel/kernel_path_branched.h"
#include "kernel/kernel_bake.h"

#define thread_num 64

CCL_NAMESPACE_BEGIN

extern "C" void* ldm_malloc(size_t size);
extern "C" void ldm_free(void *addr, size_t size);

__thread_local int row;

__thread_local crts_rply_t dma_rply = 0;

__thread_local unsigned int D_COUNT = 0;

typedef struct Para{
    KernelGlobals *kg;
    float *buffer;
    int sample;
    int x;
    int y;
    int h;
    int w;
    int offset;
    int stride;
}Para;


extern "C" void func(Para* master){

    Para slave;

    KernelGlobals *kg =  (KernelGlobals*)ldm_malloc(sizeof(KernelGlobals));

    CRTS_dma_iget(&slave,master,sizeof(Para),&dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply,D_COUNT);

    CRTS_dma_iget(kg,master->kg,sizeof(KernelGlobals),&dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply,D_COUNT);

    CRTS_dma_iget(&slave,master,sizeof(Para),&dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply,D_COUNT);

    float *buffer =  (float*)ldm_malloc(sizeof(float)*slave.w*kernel_data.film.pass_stride);

    row=_PEN;
    
    for (int y = slave.y+_PEN; y < slave.y + slave.h; y+=thread_num, row+=thread_num) {

      CRTS_dma_iget( buffer,master->buffer+row*kernel_data.film.pass_stride* slave.w, slave.w*kernel_data.film.pass_stride*4,&dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply,D_COUNT);

      for (int x = slave.x; x < slave.x + slave.w; x++) {
        kernel_path_trace(kg, &buffer[(x% slave.w)*kernel_data.film.pass_stride], slave.sample, x, y, slave.offset, slave.stride);
      }

      CRTS_dma_iput(master->buffer+row*kernel_data.film.pass_stride* slave.w, buffer, slave.w*kernel_data.film.pass_stride*4,&dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply,D_COUNT);

    }

  ldm_free(kg,sizeof(KernelGlobals));
  ldm_free(buffer,sizeof(float)*slave.w*kernel_data.film.pass_stride);



}

CCL_NAMESPACE_END
