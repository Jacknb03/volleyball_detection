#!/bin/bash
# 检查 RealSense 是否被系统识别（插着相机运行）
echo "=== USB 设备（找 Intel 8086）==="
lsusb | grep -i "8086\|Intel" || echo "❌ 未检测到 Intel RealSense（相机未被 USB 识别）"

echo ""
echo "=== 已装 RealSense 软件 ==="
dpkg -l | grep -i realsense | awk '{print $2, $3}' || true

echo ""
echo "=== 最近 USB 内核日志 ==="
dmesg 2>/dev/null | tail -30 | grep -iE "usb|intel|8086" || journalctl -k -n 20 --no-pager 2>/dev/null | grep -iE "usb|intel"

echo ""
echo "若上面显示「未检测到」："
echo "  1. 换一根支持数据传输的 USB-C 线（很多线只能充电）"
echo "  2. 插电脑上的 USB3 口（蓝色/SS 标记），别用 Hub"
echo "  3. 拔插后运行: lsusb | grep 8086"
echo "  4. 正常应看到类似: Bus ... ID 8086:0b5c Intel Corp. ..."
