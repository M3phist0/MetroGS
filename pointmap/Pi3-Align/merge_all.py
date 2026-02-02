import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os
import glob # 导入 glob 模块用于文件模式匹配

def merge_multiple_ply_files(file_paths, output_path):
    """
    读取多个PLY文件，提取所有文件的xyz和rgb数据，并合并成一个新文件。

    Args:
        file_paths (list): 包含所有PLY文件路径的列表。
        output_path (str): 合并后PLY文件的保存路径。
    """
    if not file_paths:
        print("错误: 未提供任何输入文件路径。")
        return

    # 存储所有文件的 x, y, z, r, g, b 数据的列表
    x_list, y_list, z_list = [], [], []
    r_list, g_list, b_list = [], [], []
    total_points = 0
    
    # 尝试确定颜色属性名 (默认使用 'red', 'green', 'blue')
    color_attrs = ('red', 'green', 'blue')

    print(f"--- 开始读取并合并 {len(file_paths)} 个 PLY 文件 ---")
    
    for i, file_path in enumerate(file_paths):
        print(f"正在处理文件 {i+1}/{len(file_paths)}: {file_path}")
        
        try:
            ply = PlyData.read(file_path)
            v_data = ply['vertex']
            
            # 检查并更新颜色属性名（只需检查第一个文件）
            # if i == 0 and 'r' in v_data.dtype.names:
            #     color_attrs = ('r', 'g', 'b')

            # 提取 XYZ
            x_list.append(v_data['x'])
            y_list.append(v_data['y'])
            z_list.append(v_data['z'])

            # 提取 RGB
            r_list.append(v_data[color_attrs[0]])
            g_list.append(v_data[color_attrs[1]])
            b_list.append(v_data[color_attrs[2]])

            total_points += len(v_data)
            
        except FileNotFoundError:
            print(f"警告: 文件未找到，跳过: {file_path}")
        except KeyError as e:
            print(f"警告: 文件 {file_path} 缺少必要的属性，跳过。缺失属性: {e}")
        except Exception as e:
            print(f"警告: 读取文件 {file_path} 时发生未知错误，跳过: {e}")
            
    if total_points == 0:
        print("错误: 所有输入文件均无效或点数为零，无法合并。")
        return

    # --- 3. 最终合并数据 ---
    print(f"所有文件读取完成。总点数: {total_points}。开始合并...")
    
    x_merged = np.hstack(x_list)
    y_merged = np.hstack(y_list)
    z_merged = np.hstack(z_list)
    
    # 确保颜色数据类型为 uint8 (u1)
    r_merged = np.hstack(r_list).astype(np.uint8)
    g_merged = np.hstack(g_list).astype(np.uint8)
    b_merged = np.hstack(b_list).astype(np.uint8)
    
    # --- 4. 构建新的PlyData对象 ---
    # 使用 'red', 'green', 'blue' 作为输出的统一命名
    merged_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    merged_data = np.zeros(total_points, dtype=merged_dtype)
    
    merged_data['x'] = x_merged
    merged_data['y'] = y_merged
    merged_data['z'] = z_merged
    merged_data['red'] = r_merged
    merged_data['green'] = g_merged
    merged_data['blue'] = b_merged

    # --- 5. 写入新文件 ---
    el = PlyElement.describe(merged_data, 'vertex')
    PlyData([el], text=False).write(output_path)
    
    print(f"\n--- 成功合并！---")
    print(f"合并后的文件已保存到: {output_path}")

def parse_args():
    """
    解析命令行参数，获取基础路径和输出文件路径。
    """
    parser = argparse.ArgumentParser(description="自动搜索指定目录下所有 Block 的 PLY 文件并合并。")
    # 关键修改：接收一个 base_dir 参数
    parser.add_argument(
        '--base_dir', 
        type=str, 
        required=True,
        help='包含 segments/block_i/... 结构的基础目录路径。'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='merged_all.ply',
        help='合并后PLY文件的保存路径。默认为 "merged_all.ply"。'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # --- 自动搜索 PLY 文件 ---
    
    # 构造搜索模式
    # 路径格式: {base_dir}/segments/block_*/output/pcd/combined_pcd.ply
    search_pattern = os.path.join(
        args.base_dir, 
        "segments", 
        "block_*", 
        "output", 
        "pcd", 
        "combined_pcd.ply"
    )

    # 使用 glob 查找所有匹配的文件路径
    ply_files = glob.glob(search_pattern)

    print(f"在基础目录 '{args.base_dir}' 下找到以下 {len(ply_files)} 个 PLY 文件:")
    for file_path in ply_files:
        print(f" - {file_path}")
    print("-" * 30)

    # 调用合并函数
    if len(ply_files) > 0:
        merge_multiple_ply_files(ply_files, args.output)
    else:
        print("错误: 未找到任何匹配的 PLY 文件。请检查 base_dir 或文件结构。")