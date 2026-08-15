"""系统资源读取：优先读 /proc（Linux，无第三方依赖），失败回退 psutil。"""
import time


def _read_proc_stat():
    with open('/proc/stat', 'r') as f:
        parts = f.readline().split()
    # cpu user nice system idle iowait irq softirq steal ...
    if not parts or parts[0] != 'cpu' or len(parts) < 5:
        raise ValueError('bad /proc/stat')
    idle = int(parts[4])
    total = sum(int(x) for x in parts[1:])
    return total, idle


def cpu_percent(interval=0.2):
    """返回 CPU 占用百分比（0-100），失败返回 None。"""
    try:
        t1, i1 = _read_proc_stat()
        time.sleep(max(interval, 0.05))
        t2, i2 = _read_proc_stat()
        dt = t2 - t1
        di = i2 - i1
        if dt <= 0:
            return 0.0
        return round((1 - di / dt) * 100, 1)
    except Exception:
        pass
    # 回退 psutil（Windows 等无 /proc 环境）
    try:
        import psutil
        return psutil.cpu_percent(interval=interval)
    except Exception:
        return None


def mem_percent():
    """返回内存占用百分比（0-100），失败返回 None。"""
    try:
        d = {}
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                parts = line.split(':')
                if len(parts) == 2:
                    d[parts[0].strip()] = int(parts[1].split()[0])
        total = d.get('MemTotal', 0)
        avail = d.get('MemAvailable', d.get('MemFree', 0))
        if not total:
            return None
        return round((1 - avail / total) * 100, 1)
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().percent
    except Exception:
        return None
