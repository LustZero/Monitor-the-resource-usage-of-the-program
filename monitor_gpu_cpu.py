import psutil          # 用于获取进程CPU、内存等系统资源信息
import time            # 用于控制监控间隔时间
import os              # 用于获取当前进程PID
import datetime        # 用于记录时间戳
import pynvml          # NVIDIA GPU监控库，用于读取GPU显存和利用率


def monitor_process(pid=None):
    """
    监控指定进程的 CPU、内存、GPU 使用情况，并持续写入日志文件
    :param pid: 要监控的进程ID，如果为空则默认监控当前Python进程
    """

    # 如果未指定 PID，则默认监控当前运行脚本自身
    if pid is None:
        pid = os.getpid()

    # 保存每次采样的 GPU 显存占用（当前进程）
    gpu_list = []

    # 保存每次采样的 CPU 内存 RSS（实际物理内存占用）
    cpu_list = []

    # ----------------------------
    # 初始化 GPU 监控模块 NVML
    # ----------------------------
    try:
        pynvml.nvmlInit()   # 初始化 NVIDIA 管理库
        gpu_device_count = pynvml.nvmlDeviceGetCount()  # 获取GPU数量
        print(f"检测到 {gpu_device_count} 个GPU设备")
    except Exception as e:
        print(f"GPU监控初始化失败: {e}")
        gpu_device_count = 0

    try:
        # 创建日志文件，并写入开始标记
        with open('./monitor_record.txt', 'w') as f:
            f.write("start!\n")

        # 获取目标进程对象
        p = psutil.Process(pid)
        print(f"Monitoring process: {p.name()} (PID: {p.pid})")

        # 当进程仍在运行时持续监控
        while p.is_running():

            # ----------------------------
            # 获取 CPU 和内存信息
            # ----------------------------

            # 获取当前进程CPU占用率 (%)
            cpu_percent = p.cpu_percent()

            # 获取内存详细信息
            mem_info = p.memory_info()

            # 获取内存占系统总内存百分比
            mem_percent = p.memory_percent()

            # ----------------------------
            # 获取 GPU 信息
            # ----------------------------
            gpu_info = []

            if gpu_device_count > 0:
                for i in range(gpu_device_count):
                    try:
                        # 获取第 i 块GPU句柄
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                        # GPU显存信息
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        # GPU利用率信息
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                        # 当前进程在该GPU上的显存使用量
                        process_gpu_memory = 0

                        # 获取所有正在使用该GPU的计算进程
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

                        for proc in processes:
                            if proc.pid == pid:
                                # 当前进程GPU显存(MB)
                                process_gpu_memory = proc.usedGpuMemory / (1024 * 1024)

                                # 保存用于后续统计最大值
                                gpu_list.append(process_gpu_memory)

                                # 保存当前GPU信息
                                gpu_info.append({
                                    'gpu_id': i,
                                    'total_memory': memory_info.total / (1024 * 1024),   # 总显存 MB
                                    'used_memory': memory_info.used / (1024 * 1024),     # 已使用显存 MB
                                    'process_memory': process_gpu_memory,                 # 当前进程显存 MB
                                    'gpu_utilization': utilization.gpu,                   # GPU利用率 %
                                    'memory_utilization': utilization.memory              # 显存带宽利用率 %
                                })

                    except Exception as e:
                        print(f"获取GPU {i} 信息失败: {e}")
                        continue

            # ----------------------------
            # 保存 CPU 内存占用记录
            # ----------------------------
            cpu_list.append(mem_info.rss / 1024 / 1024)

            # ----------------------------
            # 写入日志文件
            # ----------------------------
            with open('./monitor_record.txt', 'a') as f:
                # 当前时间戳
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

                f.write(timestamp + '\n')

                # 写入CPU和内存信息
                f.write(
                    f"PID {p.pid}: CPU {cpu_percent:5.1f}% | "
                    f"Memory RSS {mem_info.rss/1024/1024:7.2f} MB ({mem_percent:.1f}%)\n"
                )

                # 写入GPU信息
                if gpu_info:
                    for gpu in gpu_info:
                        f.write(
                            f"GPU {gpu['gpu_id']}: Total {gpu['total_memory']:7.1f} MB | "
                            f"Used {gpu['used_memory']:7.1f} MB | "
                            f"Process {gpu['process_memory']:7.1f} MB | "
                            f"GPU Util {gpu['gpu_utilization']:5.1f}% | "
                            f"Mem Util {gpu['memory_utilization']:5.1f}%\n"
                        )
                else:
                    f.write("GPU: No GPU information available\n")

                # 空行分隔每次记录
                f.write("\n")

            # ----------------------------
            # 控制台打印信息
            # ----------------------------
            print(timestamp)
            print(
                f"PID {p.pid}: CPU {cpu_percent:5.1f}% | "
                f"Memory RSS {mem_info.rss/1024/1024:7.2f} MB ({mem_percent:.1f}%)"
            )

            if gpu_info:
                for gpu in gpu_info:
                    print(
                        f"GPU {gpu['gpu_id']}: Total {gpu['total_memory']:7.1f} MB | "
                        f"Used {gpu['used_memory']:7.1f} MB | "
                        f"Process {gpu['process_memory']:7.1f} MB | "
                        f"GPU Util {gpu['gpu_utilization']:5.1f}% | "
                        f"Mem Util {gpu['memory_utilization']:5.1f}%"
                    )

            print("-" * 80)

            # 每2秒采样一次
            time.sleep(2)

    # ----------------------------
    # 异常处理
    # ----------------------------
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} no longer exists.")

    except Exception as e:
        print(f"监控过程中发生错误: {e}")

    finally:
        # 程序结束时关闭 NVML
        if gpu_device_count > 0:
            pynvml.nvmlShutdown()

        # 写入结束标记
        with open('./monitor_record.txt', 'a') as f:
            f.write("end!")

    # ----------------------------
    # 统计最大值
    # ----------------------------
    cpu_max = max(cpu_list)
    gpu_max = max(gpu_list)

    # 写入最大值结果
    with open('./monitor_record.txt', 'a') as f:
        f.write(f"cpu_max:{cpu_max}, gpu_max:{gpu_max}")


# --------------------------------------
# 监控指定 PID 的进程
# --------------------------------------
monitor_process(1090082)
