#include <crts.h>
#include <slave.h>
#include "slave_math.h"
#include <athread.h>
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

#include "util/util_atomic.h"
#include "util/util_math_matrix.h"
#include "kernel/filter/filter_defines.h"
#include "kernel/filter/filter_features.h"
#include "kernel/filter/filter_reconstruction.h"
#include "kernel/filter/filter_nlm_cpu.h"

#define thread_num 64

CCL_NAMESPACE_BEGIN

extern "C" void* ldm_malloc(size_t size);
extern "C" void ldm_free(void *addr, size_t size);

__thread_local int row;

__thread_local crts_rply_t dma_rply = 0;

__thread_local unsigned int D_COUNT = 0;

__thread_local_share float buffer1[4000]={0};

typedef struct Para{
    KernelGlobals *kg;
    float *buffer;
    int proc_id;
    int sample;
    int x;
    int y;
    int h;
    int w;
    int offset;
    int stride;
    int start_sample;
    int end_sample;
}Para;

     typedef struct Para_lm{
       int r;
       int f;
       int4 rect;
       int w;
       int h;
       float *guide_p;
       float *variance_p;
       float *image_p;
       float *weightAccum_p;
       float *out_p;
       int stride;
       int channel_offset;
       int k_2;
       int a;
     }Para_lm;

    typedef struct Para_den{
       int r;
       int task_rs_source_w;
       int task_rs_source_h;
       float *color_p;
       float *color_var_p;
       float *scale_p;
       int stride;
       int channel_offset;
       int frame_offset;
       float k_2;
       int t;
       float *buffer;
       float *transform;
       int *rank;
       float *XtWX;
       float3 *XtWY;
       int *filter_window;
       bool use_time;
    }Para_den;


typedef unordered_map<float, float> CoverageMap;



extern "C" void func(Para* master){

    Para slave;

    KernelGlobals *kg =  (KernelGlobals*)ldm_malloc(sizeof(KernelGlobals));

    CRTS_dma_iget(&slave,master,sizeof(Para),&dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply,D_COUNT);

    CRTS_dma_iget(kg,master->kg,sizeof(KernelGlobals),&dma_rply);
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



/*

///* 6cgps   1master+384slaves edition///////
extern "C" void func(Para* master){

    Para slave;

    KernelGlobals *kg =  (KernelGlobals*)ldm_malloc(sizeof(KernelGlobals));
    

    CRTS_dma_iget(&slave,master,sizeof(Para),&dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply,D_COUNT);

    CRTS_dma_iget(kg,master->kg,sizeof(KernelGlobals),&dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply,D_COUNT);


    CoverageMap *coverage_m  =  (CoverageMap*)ldm_malloc(sizeof(CoverageMap)*4000);
    CoverageMap *coverage_obj  =  (CoverageMap*)ldm_malloc(sizeof(CoverageMap)*4000);
    CoverageMap *coverage_as  =  (CoverageMap*)ldm_malloc(sizeof(CoverageMap)*4000);
    kg->coverage_material=coverage_m;
    kg->coverage_object=coverage_obj;
    kg->coverage_asset=coverage_as;

    int average = slave.h*slave.w/64;
    int num_tasks = average;
    int offset;
    int remainder  =  slave.h*slave.w%64;
    int myid = _PEN ;
//    int myid = _MYID;
    if(myid < remainder) {
        num_tasks++;
        offset = myid*num_tasks;
    }
    else{
        offset = myid*num_tasks+remainder;
    }
    float *buffer =  (float*)ldm_malloc(sizeof(float)*num_tasks*kernel_data.film.pass_stride);
    for(int i=0;i<num_tasks*kernel_data.film.pass_stride;i++)
         buffer[i]=0.0f;
    int y = offset/slave.w;
    int x = offset%slave.w;

    for(int i = 0; i < num_tasks;++i){
//        printf("_MYID %d ,offset %d,x %d, y %d,num_tasks %d\n",_MYID,offset,x%slave.w,y+x/slave.w,num_tasks);
        for(int sample = 0; sample < 128;++sample){
            kernel_path_trace(kg, &buffer[i*kernel_data.film.pass_stride], sample, x%slave.w+slave.x, y+x/slave.w+slave.y, slave.offset, slave.stride);
        }
        x++;
    }

    CRTS_dma_iput( master->buffer+(offset*kernel_data.film.pass_stride), buffer, num_tasks*kernel_data.film.pass_stride*4,&dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply,D_COUNT);
    ldm_free(kg,sizeof(KernelGlobals));
    ldm_free(buffer,sizeof(float)*num_tasks*kernel_data.film.pass_stride);



}

*/
extern "C" void func_lm(Para* master_lm){

   Para_lm pl;
   CRTS_dma_iget(&pl,master_lm,sizeof(Para_lm),&dma_rply);
   D_COUNT++;
   CRTS_dma_wait_value(&dma_rply,D_COUNT);

   int r=pl.r;
   float slave_diff[4096]={0};
   float slave_blur[4096]={0};

   float slave_weight[4096]={0};
   float slave_out[4096]={0};

   //CRTS_memcpy_sldm(buffer1,pl.guide_p,4*4,MEM_TO_LDM);

    int end = (2 * r + 1) * (2 * r + 1);
    int range = (2 * r + 1);
    for (int i =_MYID ; i < end; i+=64) {
      int dy = i / range - r;
      int dx = i % range - r;


      int local_rect[4] = {
          max(0, -dx), max(0, -dy), pl.rect.z - pl.rect.x - max(0, dx), pl.rect.w - pl.rect.y - max(0, dy)};
      kernel_filter_nlm_calc_difference(dx,
                                          dy,
                                          (float *)pl.guide_p,
                                          (float *)pl.variance_p,
                                          NULL,
                                          slave_diff,
                                          load_int4(local_rect),
                                          pl.w,
                                          pl.channel_offset,
                                          0,
                                          pl.a,
                                          pl.k_2);

      kernel_filter_nlm_blur(slave_diff,slave_blur, load_int4(local_rect), pl.w, pl.f);
      kernel_filter_nlm_calc_weight(slave_blur, slave_diff, load_int4(local_rect), pl.w, pl.f);
      kernel_filter_nlm_blur(slave_diff, slave_blur, load_int4(local_rect), pl.w, pl.f);

      kernel_filter_nlm_update_output(dx,
                                        dy,
                                        slave_blur,
                                        (float *)pl.image_p,
                                        slave_diff,
                                        (float *)slave_out,
                                        (float *)slave_weight,
                                        load_int4(local_rect),
                                        pl.channel_offset,
                                        pl.stride,
                                        pl.f);
      }

     

      CRTS_dma_iput(pl.weightAccum_p, slave_weight,pl.w*pl.h*4,&dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply,D_COUNT);
      
      CRTS_dma_iput(pl.out_p, slave_out, pl.w*pl.h*4,&dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply,D_COUNT);

}

extern "C" void func1(Para* master_den){

   Para_den pd;
   CRTS_dma_iget(&pd,master_den,sizeof(Para_den),&dma_rply);
   D_COUNT++;
   CRTS_dma_wait_value(&dma_rply,D_COUNT);

   int r=pd.r;

    float slave_diff[6000]={0};
    float slave_blur[6000]={0};
    for (int i =_MYID ; i < (2 * r + 1) * (2 * r + 1); i+=64) {
      int dy = i / (2 * r + 1) - r;
      int dx = i % (2 * r + 1) - r;

      int local_rect[4] = {max(0, -dx),max(0, -dy),
                           pd.task_rs_source_w - max(0, dx),
                           pd.task_rs_source_h - max(0, dy)};

      kernel_filter_nlm_calc_difference(dx,dy,(float *)pd.color_p,(float *)pd.color_var_p,(float *)pd.scale_p,
                                          (float *)slave_diff,load_int4(local_rect),pd.stride,pd.channel_offset,
                                          pd.frame_offset,1.0f,pd.k_2);
      kernel_filter_nlm_blur(slave_diff, slave_blur, load_int4(local_rect), pd.stride, 4);
      kernel_filter_nlm_calc_weight(
             slave_blur,  slave_diff, load_int4(local_rect), pd.stride, 4);
      kernel_filter_nlm_blur(slave_diff, slave_blur, load_int4(local_rect), pd.stride, 4);
      kernel_filter_nlm_construct_gramian(dx,dy,pd.t,
                                            slave_blur,
                                            (float *)pd.buffer,
                                            (float *)pd.transform,
                                            (int *)pd.rank,
                                            (float *)pd.XtWX+4096*78*_MYID,
                                            (float3 *)pd.XtWY+4096*12*_MYID,
                                            load_int4(local_rect),
                                            load_int4(pd.filter_window),
                                            pd.stride,
                                            4,
                                            pd.channel_offset,
                                            pd.frame_offset,
                                            pd.use_time);
     
      }

}

extern "C" void func_reduce(Para* master_den){

   Para_den pd;
   CRTS_dma_iget(&pd,master_den,sizeof(Para_den),&dma_rply);
   D_COUNT++;
   CRTS_dma_wait_value(&dma_rply,D_COUNT);

   int core_block_X=4096*78;
   int core_block_Y=4096*12*4;
   int X_block=4992;// 4096*78/64
   int Y_block=3072;// 4096*12*4/64
   float result_X[4992];
   float result_Y[3072];
  // int X_block=7956;// 4096*78/64
  // int Y_block=4896;// 4096*12*4/64
   //float result_X[7956];
   //float result_Y[4896];

   CRTS_dma_iget(result_X,pd.XtWX+_MYID*X_block,X_block*4,&dma_rply);
   D_COUNT++;
   CRTS_dma_wait_value(&dma_rply,D_COUNT);
   
   float *Y_p=(float *)pd.XtWY; 
   CRTS_dma_iget(result_Y,Y_p+_MYID*Y_block,Y_block*4,&dma_rply);
   D_COUNT++;
   CRTS_dma_wait_value(&dma_rply,D_COUNT);
   float num_X[4992];
   float num_Y[3072];
   //for(int i=1;i<64;i++)
   for(int i=1;i<64;i++)
   {
        
       CRTS_dma_iget(num_X,pd.XtWX+(core_block_X*i)+_MYID*X_block,X_block*4,&dma_rply);
       D_COUNT++;
       CRTS_dma_wait_value(&dma_rply,D_COUNT);
       
       CRTS_dma_iget(num_Y,Y_p+(core_block_Y*i)+_MYID*Y_block,Y_block*4,&dma_rply);
       D_COUNT++;
       CRTS_dma_wait_value(&dma_rply,D_COUNT);
       
       for(int j=0;j<X_block;j++)
          result_X[j]+=num_X[j];   
       for(int j=0;j<Y_block;j++)
          result_Y[j]+=num_Y[j];   

   }
   
      CRTS_dma_iput(pd.XtWX+_MYID*X_block, result_X, X_block*4,&dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply,D_COUNT);

      CRTS_dma_iput(Y_p+_MYID*Y_block, result_Y, Y_block*4,&dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply,D_COUNT);

}

/*
extern "C" void func_reduce2(Para* master_den){

   Para_den pd;
   CRTS_dma_iget(&pd,master_den,sizeof(Para_den),&dma_rply);
   D_COUNT++;
   CRTS_dma_wait_value(&dma_rply,D_COUNT);

   int core_block_X=4096*78;
   int core_block_Y=4096*12*4;
   int X_block=4992;// 4096*78/64
   int Y_block=3072;// 4096*12*4/64
   float result_X[4992];
   float result_Y[3072];

   CRTS_dma_iget(result_X,pd.XtWX+_MYID*X_block,X_block*4,&dma_rply);
   D_COUNT++;
   CRTS_dma_wait_value(&dma_rply,D_COUNT);
   
   float *Y_p=(float *)pd.XtWY; 
   CRTS_dma_iget(result_Y,Y_p+_MYID*Y_block,Y_block*4,&dma_rply);
   D_COUNT++;
   CRTS_dma_wait_value(&dma_rply,D_COUNT);
   float num_X[4992];
   float num_Y[3072];
   for(int i=1;i<64;i++)
   {
        
       CRTS_dma_iget(num_X,pd.XtWX+(core_block_X*i)+_MYID*X_block,X_block*4,&dma_rply);
       D_COUNT++;
       CRTS_dma_wait_value(&dma_rply,D_COUNT);
       
       CRTS_dma_iget(num_Y,Y_p+(core_block_Y*i)+_MYID*Y_block,Y_block*4,&dma_rply);
       D_COUNT++;
       CRTS_dma_wait_value(&dma_rply,D_COUNT);
       
       for(int j=0;j<X_block;j++)
          result_X[j]+=num_X[j];   
       for(int j=0;j<Y_block;j++)
          result_Y[j]+=num_Y[j];   

   }
   
      CRTS_dma_iput(pd.XtWX+_MYID*X_block, result_X, X_block*4,&dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply,D_COUNT);

      CRTS_dma_iput(Y_p+_MYID*Y_block, result_Y, Y_block*4,&dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply,D_COUNT);

}*/
CCL_NAMESPACE_END
