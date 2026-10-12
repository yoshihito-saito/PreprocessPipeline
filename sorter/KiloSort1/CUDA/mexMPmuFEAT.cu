/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "ks_cuda_checks.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
using namespace std;

const int  Nthreads = 1024,  NchanMax = 128, block = 32, NrankMax = 3;
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){    
  volatile __shared__ float  sW[81*NrankMax], sdata[(Nthreads+81)*NrankMax]; 
  float x;
  int tid, tid0, bid, i, nid, Nrank, NT, Nfilt, nt0;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  Nfilt    	=   (int) Params[1];
  NT      	=   (int) Params[0];
  Nrank     = (int) Params[6];
  nt0       = (int) Params[9];
  
  if(tid<nt0*((int) Params[6]))
      sW[tid]= W[tid%nt0 + (bid + Nfilt * (tid/nt0))* nt0];
  __syncthreads();
  
  tid0 = 0;
  while (tid0<NT-Nthreads-nt0+1){
	  if (tid<nt0*NrankMax) sdata[tid%nt0 + (tid/nt0)*(Nthreads+nt0)] = 
			data[tid0 + tid%nt0+ NT*(bid + Nfilt*(tid/nt0))];
	  #pragma unroll 3
      for(nid=0;nid<Nrank;nid++){
          sdata[tid + nt0+nid*(Nthreads+nt0)] = data[nt0+tid0 + tid+ NT*(bid +nid*Nfilt)];
	  }
	  __syncthreads();
      
	  x = 0.0f;
      for(nid=0;nid<Nrank;nid++){
		  #pragma unroll 4
          for(i=0;i<nt0;i++)
              x    += sW[i + nid*nt0] * sdata[i+tid + nid*(Nthreads+nt0)];
	  }
      conv_sig[tid0  + tid + NT*bid]   = x;
      
      tid0+=Nthreads;
      __syncthreads();
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  bestFilter(const double *Params, const float *data, 
	const float *mu, const float *lam, const float *nu, float *xbest, float *err, int *ftype){
  int tid, tid0, i, bid, NT, Nfilt, ibest = 0;
  float Th,  Cf, Ci, xb, Cbest = 0.0f, epu, cdiff;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT 		= (int) Params[0];
  Nfilt 	= (int) Params[1];
  Th 		= (float) Params[2];
  epu       = (float) Params[8];
 
  tid0 = tid + bid * Nthreads;
  if (tid0<NT & tid0>0){
      for (i=0; i<Nfilt;i++){
          Ci = data[tid0 + NT * i] + mu[i] * lam[i];
          Cf = Ci * Ci / (lam[i] + 1.0f) - lam[i]*mu[i]*mu[i];
          // add the shift component
          cdiff = data[tid0+1 + NT * i] - data[tid0-1 + NT * i];
          Cf = Cf + cdiff * cdiff / (epu + nu[i]);

		if (Cf > Cbest + 1e-6){
			Cbest 	= Cf;
			xb      = Ci - lam[i] * mu[i]; // /(lam[i] + 1);
			ibest 	= i;
		}
    }
    if (Cbest > Th*Th){
      err[tid0] 	= Cbest;
      xbest[tid0] 	= xb;
      ftype[tid0] 	= ibest;
    }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	cleanup_spikes(const double *Params, const float *xbest, const float *err, 
	const int *ftype, int *st, int *id, float *x, float *C, int *counter){
  int lockout, indx, maxFR, NTOT, tid, bid, NT, tid0,  j;
  volatile __shared__ float sdata[Nthreads+2*81+1];
  bool flag=0;
  float err0;
  
  lockout   = (int) Params[9] - 1;
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  
  NT      	=   (int) Params[0];
  maxFR 	= (int) Params[3];
  tid0 		= bid * Nthreads;


  if(tid0<NT-Nthreads-lockout+1){       
    if (tid<2*lockout)
      sdata[tid] = err[tid0 + tid];
    sdata[tid+2*lockout] = err[2*lockout + tid0 + tid];

    __syncthreads();
    
    err0 = sdata[tid+lockout];
    if(err0>1e-10){
        flag = 0;
        for(j=-lockout;j<=lockout;j++)
            if(sdata[tid+lockout+j]>err0){
                flag = 1;
                break;
            }
        if(flag==0){
            indx = atomicAdd(&counter[0], 1);
            if (indx<maxFR){
                st[indx] = tid+lockout         + tid0;
                id[indx] = ftype[tid+lockout   + tid0];
                x[indx]  = xbest[tid+lockout     + tid0];
                C[indx]  = err0;
            }
        }
    }
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	extractFEAT(const double *Params, const int *st, const int *id, 
        const float *x, const int *counter, const float *dout, const float *WtW, 
        const float *lam, const float *mu, float *d_feat){
  int t, tid, bid,  NT, ind, tcurr, Nfilt;
  float rMax, Ci, Cf;
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT 		= (int) Params[0];
  Nfilt 	= (int) Params[1];

//  ind = bid;
  
  while (tid<Nfilt){
    for(ind=counter[1]+bid;ind<counter[0];ind+=Nfilt){
      tcurr = st[ind];
      rMax = 0.0f;
      //rMax = dout[tcurr + tid*NT];
      for (t=-3;t<3;t++)
         rMax = max(rMax, dout[tcurr +t+ tid*NT]);
      
    //  Ci = dout[tcurr + tid*NT] + mu[tid] * lam[tid];
    //  Cf = Ci * Ci / (lam[tid] + 1.0f) - mu[tid]*mu[tid] * lam[tid];
      
      d_feat[tid + ind * Nfilt] = rMax;
          
      //d_feat[tid + ind * Nfilt] = dout[tcurr + tid*NT];
                //+ x[ind] * WtW[nt0 + id[ind]*(2*nt0-1) + (2*nt0-1)*Nfilt*tid];
      //ind += Nfilt;
    }
    tid += Nthreads;
  }
 }
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	subSpikes(const double *Params, const int *st, const int *id, 
        const float *x, const int *counter, float *dout, const float *WtW){
  int nt0, tid, bid,  NT, ind, tcurr, Nfilt;
  
 nt0       = (int) Params[9];
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT 		= (int) Params[0];
  Nfilt 	= (int) Params[1];

  for(ind=counter[1]; ind<counter[0];ind++){
    tcurr = tid + st[ind]-nt0+1;
    if (tcurr>=0 & tcurr<NT)
      dout[tcurr + bid*NT] -= x[ind] * WtW[tid + id[ind]*(2*nt0-1) + (2*nt0-1)*Nfilt*bid];
  }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	subtract_spikes(const double *Params,  const int *st, 
        const int *id, const float *x, const int *counter, float *dataraw, 
        const float *W, const float *U){
  int nt0, tid, bid, Nblocks, i, NT, ind, Nchan;
  __shared__ float sh_W[81], sh_U[NchanMax];
  
  nt0       = (int) Params[9];
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  Nblocks   = gridDim.x;
  NT        = (int) Params[0];
  Nchan     = (int) Params[5];
  ind       = bid;

  while(ind<counter[0]){
    if (tid<nt0) sh_W[tid] = W[tid + nt0*id[ind]];
    sh_U[tid] = U[tid + Nchan*id[ind]];

    __syncthreads();
    for (i=0;i<nt0;i++)
      dataraw[i + st[ind] + NT * tid] -= x[ind] * sh_W[i] * sh_U[tid];
    ind+= Nblocks;
    __syncthreads();
  }

}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void getWgradient(const double *Params, const int *st, const int *id, const float *x,  const int *counter, const float *datarez, const float *U, float *dW){
  int nt0, tid, bid, i, ind, NT, Nchan;
  float xprod; 
  volatile __shared__ float sh_U[NchanMax];
  
  NT        = (int) Params[0];
  Nchan     = (int) Params[5];
  nt0       = (int) Params[9];
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  while(tid<Nchan){
    sh_U[tid] = U[tid + bid*Nchan];
    tid+= blockDim.x;
  }
  tid 		= threadIdx.x;
  __syncthreads();

  for(ind=0; ind<counter[0];ind++)
    if (id[ind]==bid){
      xprod = 0.0f;
      for (i=0;i<Nchan;i++)
	xprod+= sh_U[i] * datarez[st[ind] + tid + NT * i];
      dW[tid + nt0 * bid] += xprod * x[ind];
    }
}
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void getUgradient(const double *Params, const int *st, const int *id, const float *x,  const int *counter, const float *datarez, const float *W, float *dU){  
  int nt0, j, tid, bid, i, ind, NT, Nchan;
  float xprod; 
  volatile __shared__ float sh_M[NchanMax*81], sh_W[81];

  nt0       = (int) Params[9];
  NT = (int) Params[0];
    Nchan = (int) Params[5];

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  if (tid<nt0)
    sh_W[tid] = W[tid + nt0*bid];
 
  __syncthreads();

  for(ind=0; ind<counter[0];ind++)
    if (id[ind]==bid){
      if(tid<nt0)
	for (j=0;j<Nchan;j++)
	  sh_M[tid + nt0*j] = datarez[tid + st[ind] + NT*j];
      __syncthreads();

      xprod = 0.0f;
      for (i=0;i<nt0;i++)
	xprod+= sh_W[i] * sh_M[i + tid*nt0];
      dU[tid + bid*Nchan] += xprod * x[ind];
      __syncthreads();
    }  
}
//////////////////////////////////////////////////////////////////////////////////////////

/*
 * Host code
 */
static void ksMexImpl(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
  KsCudaResources resources;

  /* Declare input variables*/
  double *Params, *d_Params;
  int nt0, blocksPerGrid, NT, maxFR, Nchan;
  int const threadsPerBlock = Nthreads;

  /* Initialize the MathWorks GPU API. */
  if (mxInitGPU() != MX_GPU_SUCCESS) throw std::runtime_error("MathWorks GPU initialization failed.");

  /* read Params and copy to GPU */
  Params  	= (double*) mxGetData(prhs[0]);
  NT		= (int) Params[0];
  blocksPerGrid	= (int) Params[1];
  maxFR		= (int) Params[3];
  Nchan     = (int) Params[5];
  nt0       = (int) Params[9];

  KS_CUDA(resources.allocate(&d_Params,      sizeof(double)*mxGetNumberOfElements(prhs[0])));
  KS_CUDA(cudaMemcpy(d_Params,Params,sizeof(double)*mxGetNumberOfElements(prhs[0]),cudaMemcpyHostToDevice));

  /* collect input GPU variables*/
  mxGPUArray const  *W,   *data,   *WtW, *mu,   *lam, *nu;
  const float     *d_W, *d_data, *d_WtW,  *d_mu, *d_lam, *d_nu;

  data       = resources.gpu(mxGPUCreateFromMxArray(prhs[1]));
  d_data     = (float const *)(mxGPUGetDataReadOnly(data));
  W             = resources.gpu(mxGPUCreateFromMxArray(prhs[2]));
  d_W        	= (float const *)(mxGPUGetDataReadOnly(W));
  WtW       	= resources.gpu(mxGPUCreateFromMxArray(prhs[3]));
  d_WtW     	= (float const *)(mxGPUGetDataReadOnly(WtW));
  mu            = resources.gpu(mxGPUCreateFromMxArray(prhs[4]));
  d_mu          = (float const *)(mxGPUGetDataReadOnly(mu));
  lam       	= resources.gpu(mxGPUCreateFromMxArray(prhs[5]));
  d_lam     	= (float const *)(mxGPUGetDataReadOnly(lam));
  nu            = resources.gpu(mxGPUCreateFromMxArray(prhs[6]));
  d_nu          = (float const *)(mxGPUGetDataReadOnly(nu));

  /* allocate new GPU variables*/
  float *d_err,*d_C, *d_xbest, *d_x, *d_dout, *d_feat;
  int *d_st,  *d_ftype,  *d_id, *d_counter;

  KS_CUDA(resources.allocate(&d_dout,   NT * blocksPerGrid* sizeof(float)));

  KS_CUDA(resources.allocate(&d_err,   NT * sizeof(float)));
  KS_CUDA(resources.allocate(&d_xbest,   NT * sizeof(float)));
  KS_CUDA(resources.allocate(&d_ftype, NT * sizeof(int)));
  KS_CUDA(resources.allocate(&d_st,    maxFR * sizeof(int)));
  KS_CUDA(resources.allocate(&d_id,    maxFR * sizeof(int)));
  KS_CUDA(resources.allocate(&d_x,     maxFR * sizeof(float)));
  KS_CUDA(resources.allocate(&d_C,     maxFR * sizeof(float)));
  KS_CUDA(resources.allocate(&d_counter,   2*sizeof(int)));
  KS_CUDA(resources.allocate(&d_feat,     maxFR * blocksPerGrid * sizeof(float)));

  KS_CUDA(cudaMemset(d_dout,    0, NT * blocksPerGrid * sizeof(float)));
  KS_CUDA(cudaMemset(d_counter, 0, 2*sizeof(int)));
  KS_CUDA(cudaMemset(d_st,      0, maxFR *   sizeof(int)));
  KS_CUDA(cudaMemset(d_id,      0, maxFR *   sizeof(int)));
  KS_CUDA(cudaMemset(d_x,       0, maxFR *    sizeof(float)));
  KS_CUDA(cudaMemset(d_C,       0, maxFR *    sizeof(float)));
  KS_CUDA(cudaMemset(d_feat,       0, maxFR * blocksPerGrid *   sizeof(float)));

  int *counter;
  counter = resources.host((int*) calloc(1,sizeof(int)));

  // filter the data with the temporal templates
  Conv1D<<<blocksPerGrid,threadsPerBlock>>>(d_Params, d_data, d_W, d_dout);
  KS_KERNEL("Conv1D");
  for(int k=0;k<(int) Params[4];k++){
    KS_CUDA(cudaMemset(d_err,     0, NT * sizeof(float)));
    KS_CUDA(cudaMemset(d_ftype,   0, NT * sizeof(int)));
    KS_CUDA(cudaMemset(d_xbest,   0, NT * sizeof(float)));

    // compute the best filter
    bestFilter<<<NT/Nthreads,threadsPerBlock>>>(    d_Params,
            d_dout, d_mu, d_lam, d_nu, d_xbest, d_err, d_ftype);
    KS_KERNEL("bestFilter");

    // ignore peaks that are smaller than another nearby peak
    cleanup_spikes<<<NT/Nthreads,threadsPerBlock>>>(d_Params,
            d_xbest, d_err, d_ftype, d_st, d_id, d_x, d_C, d_counter);
    KS_KERNEL("cleanup_spikes");

    // add new spikes to 2nd counter
    KS_CUDA(cudaMemcpy(counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    if (counter[0]>maxFR){
      counter[0] = maxFR;
      KS_CUDA(cudaMemcpy(d_counter, counter, sizeof(int), cudaMemcpyHostToDevice));
    }

    // extract template features before subtraction
    extractFEAT<<<blocksPerGrid, threadsPerBlock>>>(d_Params, d_st, d_id,
            d_x, d_counter, d_dout, d_WtW, d_lam, d_mu,d_feat);
    KS_KERNEL("extractFEAT");

    // subtract the detected spikes
    subSpikes<<<blocksPerGrid, 2*nt0-1>>>(d_Params, d_st, d_id,
            d_x, d_counter, d_dout, d_WtW);
    KS_KERNEL("subSpikes");

    // update 1st counter from 2nd counter
    KS_CUDA(cudaMemcpy(d_counter+1, d_counter, sizeof(int), cudaMemcpyDeviceToDevice));

    if(counter[0]==maxFR)
      break;
  }

//  extractFEAT<<<blocksPerGrid, blocksPerGrid>>>(d_Params, d_st, d_id, d_x, d_counter, d_dout, d_WtW,   d_feat);

  float *x, *C, *feat;
  int *st, *id;
  int minSize;
  if (counter[0]<maxFR)  minSize = counter[0];
  else                   minSize = maxFR;
  const mwSize dimst[] 	= {minSize,1};
  plhs[0] = mxCreateNumericArray(2, dimst, mxINT32_CLASS, mxREAL);
  st = (int*) mxGetData(plhs[0]);
  plhs[1] = mxCreateNumericArray(2, dimst, mxINT32_CLASS, mxREAL);
  id = (int*) mxGetData(plhs[1]);
  plhs[2] = mxCreateNumericArray(2, dimst, mxSINGLE_CLASS, mxREAL);
  x =  (float*) mxGetData(plhs[2]);
  plhs[3] = mxCreateNumericArray(2, dimst, mxSINGLE_CLASS, mxREAL);
  C =  (float*) mxGetData(plhs[3]);

  const mwSize dimsf[] 	= {blocksPerGrid, minSize};
  plhs[4] = mxCreateNumericArray(2, dimsf, mxSINGLE_CLASS, mxREAL);
  feat =  (float*) mxGetData(plhs[4]);

  KS_CUDA(cudaMemcpy(st, d_st, minSize * sizeof(int),   cudaMemcpyDeviceToHost));
  KS_CUDA(cudaMemcpy(id, d_id, minSize * sizeof(int),   cudaMemcpyDeviceToHost));
  KS_CUDA(cudaMemcpy(x,   d_x, minSize * sizeof(float), cudaMemcpyDeviceToHost));
  KS_CUDA(cudaMemcpy(C,   d_C, minSize * sizeof(float), cudaMemcpyDeviceToHost));
  KS_CUDA(cudaMemcpy(feat,   d_feat, minSize * blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost));

  KS_CUDA(resources.releaseDevice(d_ftype));
  KS_CUDA(resources.releaseDevice(d_err));
  KS_CUDA(resources.releaseDevice(d_xbest));
  KS_CUDA(resources.releaseDevice(d_st));
  KS_CUDA(resources.releaseDevice(d_id));
  KS_CUDA(resources.releaseDevice(d_x));
  KS_CUDA(resources.releaseDevice(d_feat));
  KS_CUDA(resources.releaseDevice(d_C));
  KS_CUDA(resources.releaseDevice(d_counter));
  KS_CUDA(resources.releaseDevice(d_Params));

  KS_CUDA(resources.releaseDevice(d_dout));

  resources.releaseGpu(data);
  resources.releaseGpu(WtW);
  resources.releaseGpu(W);
  resources.releaseGpu(mu);
  resources.releaseGpu(lam);
  KS_CUDA(cudaDeviceSynchronize());
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]) {
    if (nrhs != 7 || nlhs != 5)
        mexErrMsgIdAndTxt("KiloSort:MexArguments", "mexMPmuFEAT requires 7 inputs and 5 outputs.");
    try { ksMexImpl(nlhs, plhs, nrhs, prhs); }
    catch (const KsCudaError &error) { mexErrMsgIdAndTxt("KiloSort:CUDAError", "%s", error.what()); }
    catch (const std::exception &error) { mexErrMsgIdAndTxt("KiloSort:MexRuntime", "%s", error.what()); }
}
