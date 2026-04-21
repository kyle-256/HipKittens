import torch
import tk_kernel_fwd, tk_kernel_bkwd, tk_kernel_bkwd_prep
import tk_kernel_bkwd_dq_qparallel as dqp

B,N,H,H_KV,D_QK,D_V=16,4096,64,8,192,128
dtype=torch.bfloat16

torch.manual_seed(42)
Q=torch.randn(B,N,H,D_QK,dtype=dtype,device='cuda')
K=torch.randn(B,N,H_KV,D_QK,dtype=dtype,device='cuda')
V=torch.randn(B,N,H_KV,D_V,dtype=dtype,device='cuda')
dO=torch.randn(B,N,H,D_V,dtype=dtype,device='cuda')
O=torch.zeros(B,N,H,D_V,dtype=dtype,device='cuda')
L=torch.zeros(B,H,1,N,dtype=torch.float32,device='cuda')
delta=torch.zeros(B,H,1,N,dtype=torch.float32,device='cuda')
tk_kernel_fwd.dispatch_fwd(Q,K,V,O,L)
tk_kernel_bkwd_prep.dispatch_prep(O,dO,delta)
torch.cuda.synchronize()

dQ=torch.zeros(B,H,N,D_QK,dtype=dtype,device='cuda')
dK=torch.zeros(B,N,H_KV,D_QK,dtype=dtype,device='cuda')
dV=torch.zeros(B,N,H_KV,D_V,dtype=dtype,device='cuda')

for _ in range(20):
    dQ.zero_()
    dqp.dispatch_bwd_dq(Q,K,V,dO,dQ,L,delta)
    dK.zero_(); dV.zero_()
    tk_kernel_bkwd.dispatch_bwd_combined(Q,K,V,dO,dK,dV,L,delta)
torch.cuda.synchronize()

s1=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
s1.record()
for _ in range(30):
    dQ.zero_()
    dqp.dispatch_bwd_dq(Q,K,V,dO,dQ,L,delta)
e1.record(); torch.cuda.synchronize()
dt_dq=s1.elapsed_time(e1)/30

s2=torch.cuda.Event(enable_timing=True); e2=torch.cuda.Event(enable_timing=True)
s2.record()
for _ in range(30):
    dK.zero_(); dV.zero_()
    tk_kernel_bkwd.dispatch_bwd_combined(Q,K,V,dO,dK,dV,L,delta)
e2.record(); torch.cuda.synchronize()
dt_kv=s2.elapsed_time(e2)/30

s3=torch.cuda.Event(enable_timing=True); e3=torch.cuda.Event(enable_timing=True)
s3.record()
for _ in range(30):
    dK.zero_(); dV.zero_(); dQ.zero_()
    tk_kernel_bkwd.dispatch_bwd_combined(Q,K,V,dO,dK,dV,L,delta)
    dqp.dispatch_bwd_dq(Q,K,V,dO,dQ,L,delta)
e3.record(); torch.cuda.synchronize()
dt_total=s3.elapsed_time(e3)/30

fwd_flops = 2 * B * N*N * H * (D_QK+D_V) // 2
bwd_flops = int(2.5 * fwd_flops)
tflops_total = (bwd_flops / 1e12) / (dt_total / 1e3)

print(f'=== FINAL PERFORMANCE (B={B} N={N} H={H} H_KV={H_KV}) ===')
print(f'dK+dV: {dt_kv:.3f} ms')
print(f'dQ:    {dt_dq:.3f} ms')
print(f'Sum:   {dt_kv+dt_dq:.3f} ms')
print(f'Seq:   {dt_total:.3f} ms')
print(f'BWD TFLOPS (2.5x fwd): {tflops_total:.2f}')
if tflops_total >= 1200:
    print(f'Target 1200T: PASS')
else:
    print(f'Target 1200T: FAIL (need {1200/tflops_total:.2f}x more)')
