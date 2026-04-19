import csv
import random
from datetime import datetime, timedelta


def generate_cloud_sales_data(filename="cloud_sales.csv", rows=500):
    regions = ['华北-北京', '华东-上海', '华南-广州', '西南-成都']
    instance_types = ['vGPU-RTX4090', 'vGPU-RTX3090', '通用型-8核32G', '计算型-16核64G']

    # 设置起始日期为半年前
    start_date = datetime.now() - timedelta(days=180)

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['date', 'region', 'instance_type', 'sales_volume', 'revenue'])

        for _ in range(rows):
            # 随机生成日期、区域和实例类型
            random_days = random.randint(0, 180)
            date_str = (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
            region = random.choice(regions)
            instance_type = random.choice(instance_types)

            # 随机生成销量和单价计算营收
            sales_volume = random.randint(1, 50)
            base_price = 200 if 'GPU' in instance_type else 50  # GPU实例更贵
            revenue = sales_volume * base_price * random.uniform(0.9, 1.1)  # 加入一点价格波动

            writer.writerow([date_str, region, instance_type, sales_volume, round(revenue, 2)])

    print(f"成功生成模拟数据：{filename}，共 {rows} 行。")


if __name__ == "__main__":
    generate_cloud_sales_data()