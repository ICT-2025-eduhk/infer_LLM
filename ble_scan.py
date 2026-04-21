"""BLE 设备扫描工具"""
import asyncio
from bleak import BleakScanner

async def scan():
    print('开始 BLE 扫描（10秒）...')
    devices = await BleakScanner.discover(timeout=10.0)
    print(f'\n发现 {len(devices)} 个设备:\n')
    for d in devices:
        name = d.name or '(无名称)'
        print(f'  名称: {name}')
        print(f'  地址: {d.address}')
        print()

asyncio.run(scan())
