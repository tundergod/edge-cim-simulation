/* Phase 0.3 A5 — Mali-G610 GEMM characterization (FP32 + FP16) via OpenCL.
 * Times matmul (projection) + attention bmm (native activation x activation) + a
 * square K-sweep (TFLOPS-vs-size curve, HeteroInfer Fig 1 analog). Device time from
 * profiling events; median over N iters. Emits JSON to argv[1] (or stdout).
 *
 * Build: gcc -O2 run_mali.c -o run_mali -lOpenCL
 * Run:   ./run_mali gemm.cl results.json
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TILE 16
#define WARMUP 5
#define ITERS 30
#define CK(x) do{ cl_int e=(x); if(e!=CL_SUCCESS){ fprintf(stderr,"CL err %d at %d\n",e,__LINE__); exit(1);} }while(0)

static cl_context ctx; static cl_command_queue q; static cl_kernel k32, k16;

static int cmp(const void*a,const void*b){ double d=*(const double*)a-*(const double*)b; return d<0?-1:d>0?1:0; }

/* run one (M,K,N) at given kernel/elemsize; return median device ms (or -1 on fail) */
static double run_shape(cl_kernel kern, size_t esz, int M,int K,int N){
    cl_int e;
    cl_mem A=clCreateBuffer(ctx,CL_MEM_READ_ONLY,(size_t)M*K*esz,0,&e); if(e)return -1;
    cl_mem B=clCreateBuffer(ctx,CL_MEM_READ_ONLY,(size_t)K*N*esz,0,&e); if(e){clReleaseMemObject(A);return -1;}
    cl_mem C=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,(size_t)M*N*esz,0,&e); if(e){clReleaseMemObject(A);clReleaseMemObject(B);return -1;}
    CK(clSetKernelArg(kern,0,sizeof(int),&M)); CK(clSetKernelArg(kern,1,sizeof(int),&K));
    CK(clSetKernelArg(kern,2,sizeof(int),&N)); CK(clSetKernelArg(kern,3,sizeof(cl_mem),&A));
    CK(clSetKernelArg(kern,4,sizeof(cl_mem),&B)); CK(clSetKernelArg(kern,5,sizeof(cl_mem),&C));
    size_t loc[2]={TILE,TILE};
    size_t glob[2]={((N+TILE-1)/TILE)*TILE, ((M+TILE-1)/TILE)*TILE};
    for(int i=0;i<WARMUP;i++){ CK(clEnqueueNDRangeKernel(q,kern,2,0,glob,loc,0,0,0)); }
    CK(clFinish(q));
    double ms[ITERS];
    for(int i=0;i<ITERS;i++){
        cl_event ev;
        CK(clEnqueueNDRangeKernel(q,kern,2,0,glob,loc,0,0,&ev));
        CK(clWaitForEvents(1,&ev));
        cl_ulong t0,t1;
        CK(clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,sizeof(t0),&t0,0));
        CK(clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,sizeof(t1),&t1,0));
        ms[i]=(t1-t0)/1e6;
        clReleaseEvent(ev);
    }
    clReleaseMemObject(A); clReleaseMemObject(B); clReleaseMemObject(C);
    qsort(ms,ITERS,sizeof(double),cmp);
    return ms[ITERS/2];
}

typedef struct { const char*group,*tag; int M,K,N; } Shape;

int main(int argc,char**argv){
    const char*clpath = argc>1?argv[1]:"gemm.cl";
    FILE*out = argc>2?fopen(argv[2],"w"):stdout;
    /* platform/device = first GPU */
    cl_platform_id plat; CK(clGetPlatformIDs(1,&plat,0));
    cl_device_id dev; CK(clGetDeviceIDs(plat,CL_DEVICE_TYPE_GPU,1,&dev,0));
    char dn[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(dn),dn,0);
    cl_int e; ctx=clCreateContext(0,1,&dev,0,0,&e); CK(e);
    q=clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&e); CK(e);
    FILE*f=fopen(clpath,"rb"); if(!f){fprintf(stderr,"no %s\n",clpath);return 1;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    char*src=malloc(sz+1); fread(src,1,sz,f); src[sz]=0; fclose(f);
    cl_program prog=clCreateProgramWithSource(ctx,1,(const char**)&src,0,&e); CK(e);
    if(clBuildProgram(prog,1,&dev,"",0,0)!=CL_SUCCESS){
        char buf[8192]; clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,sizeof(buf),buf,0);
        fprintf(stderr,"build log:\n%s\n",buf); return 1;
    }
    k32=clCreateKernel(prog,"gemm_f32",&e); CK(e);
    k16=clCreateKernel(prog,"gemm_f16",&e); CK(e);

    /* shape list: projections (decode M=1 + prefill), attention bmm, square K-sweep */
    Shape shapes[256]; int ns=0;
    int Hs[4]={2048,3072,4096,3584}, Fs[4]={8192,8192,14336,18944}, kvw[4]={512,1024,1024,512};
    const char*mn[4]={"1b","3b","8b","qwen"};
    for(int m=0;m<4;m++){
        shapes[ns++]=(Shape){"proj_decode",mn[m],1,Hs[m],Hs[m]};      /* q/o */
        shapes[ns++]=(Shape){"proj_decode",mn[m],1,Hs[m],kvw[m]};     /* kv */
        shapes[ns++]=(Shape){"proj_decode",mn[m],1,Hs[m],Fs[m]};      /* gate/up */
        shapes[ns++]=(Shape){"proj_decode",mn[m],1,Fs[m],Hs[m]};      /* down */
    }
    int pM[2]={128,1024};
    for(int i=0;i<2;i++){ shapes[ns++]=(Shape){"proj_prefill","8b_gate_up",pM[i],4096,14336}; }
    /* attention bmm (single head, hd=128): QK^T [seq,hd]x[hd,kv], SV [seq,kv]x[kv,hd] */
    int kvs[3]={129,513,1025};
    for(int i=0;i<3;i++){
        shapes[ns++]=(Shape){"attn","qkT_dec",1,128,kvs[i]};   /* decode QK^T */
        shapes[ns++]=(Shape){"attn","sv_dec",1,kvs[i],128};    /* decode SV */
    }
    shapes[ns++]=(Shape){"attn","qkT_pre",512,128,512};        /* prefill QK^T sample */
    int Ks[7]={64,128,256,512,1024,2048,4096};
    for(int i=0;i<7;i++){ shapes[ns++]=(Shape){"ksweep","sq",Ks[i],Ks[i],Ks[i]}; }

    fprintf(out,"{\n \"device\": \"%s\",\n \"results\": [\n",dn);
    for(int i=0;i<ns;i++){
        Shape s=shapes[i];
        double t32=run_shape(k32,4,s.M,s.K,s.N);
        double t16=run_shape(k16,2,s.M,s.K,s.N);
        double gf=2.0*s.M*s.K*s.N;
        fprintf(out,"  {\"group\":\"%s\",\"tag\":\"%s\",\"M\":%d,\"K\":%d,\"N\":%d,"
                "\"f32_ms\":%.5f,\"f16_ms\":%.5f,\"f32_gflops\":%.2f,\"f16_gflops\":%.2f}%s\n",
                s.group,s.tag,s.M,s.K,s.N,t32,t16,
                t32>0?gf/t32/1e6:0, t16>0?gf/t16/1e6:0, i<ns-1?",":"");
        fprintf(stderr,"[%d/%d] %s/%s M%dK%dN%d f32=%.3fms f16=%.3fms\n",i+1,ns,s.group,s.tag,s.M,s.K,s.N,t32,t16);
    }
    fprintf(out," ]\n}\n");
    if(out!=stdout) fclose(out);
    return 0;
}
