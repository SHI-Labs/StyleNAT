from rich import print
import torch

@torch.inference_mode()
def throughput(generator,
               rounds=100,
               warmup=10,
               batch_size=1,
               style_dim=512,
               ):
    #torch.backends.cuda.matmul.allow_tf32 = True
    times = torch.empty(rounds)
    noise = torch.randn((batch_size, style_dim), device="cuda")
    for _ in range(warmup):
        imgs, _ = generator(noise)
    for i in range(rounds):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        imgs, _ = generator(noise)
        ender.record()
        torch.cuda.synchronize()
        times[i] = starter.elapsed_time(ender)/1000
        #print(f"{imgs.shape=}")
    total_time = times.sum()
    total_images = rounds * batch_size
    imgs_per_second = total_images / total_time 
    print(f"{torch.std_mean(times)=}")
    print(f"{total_images=}, {total_time=}")
    print(f"{imgs_per_second=}")
    #print(f"{torch.std_mean(imgs_per_second)=}")
