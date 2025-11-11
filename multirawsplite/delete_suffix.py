import os
import glob
import re


def remove_frame_suffix(folder_path, dry_run=True):
    """
    去除指定文件夹中所有文件的'_frame_数字'后缀
    
    参数:
        folder_path: 要处理的文件夹路径
        dry_run: 如果为True，只显示将要重命名的文件，不实际执行；如果为False，执行重命名
    
    返回:
        重命名的文件数量
    """
    # 如果文件夹不存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在")
        return 0
    
    # 获取文件夹中所有文件
    all_files = glob.glob(os.path.join(folder_path, "*"))
    
    # 正则表达式匹配 _frame_数字 模式（在文件扩展名之前）
    pattern = re.compile(r'(.*)_frame_(\d+)(\.[^.]+)$')
    
    renamed_count = 0
    
    print(f"\n{'=' * 60}")
    print(f"处理文件夹: {folder_path}")
    print(f"模式: 干运行" if dry_run else f"模式: 实际执行")
    print(f"{'=' * 60}\n")
    
    for file_path in all_files:
        # 跳过文件夹
        if os.path.isdir(file_path):
            continue
        
        # 获取文件名（不含路径）
        file_name = os.path.basename(file_path)
        
        # 匹配后缀模式
        match = pattern.match(file_name)
        
        if match:
            # 提取基础名称和扩展名
            base_name = match.group(1)
            frame_num = match.group(2)
            extension = match.group(3)
            
            # 构建新文件名
            new_file_name = f"{base_name}{extension}"
            new_file_path = os.path.join(folder_path, new_file_name)
            
            # 检查目标文件是否已存在
            if os.path.exists(new_file_path) and new_file_path != file_path:
                print(f"⚠ 跳过（目标文件已存在）: {file_name}")
                continue
            
            print(f"{'[预览]' if dry_run else '[执行]'} {file_name}")
            print(f"      → {new_file_name}")
            
            # 如果不是干运行，执行重命名
            if not dry_run:
                try:
                    os.rename(file_path, new_file_path)
                    print(f"      ✓ 重命名成功")
                except Exception as e:
                    print(f"      ✗ 重命名失败: {e}")
                    continue
            
            renamed_count += 1
            print()
    
    print(f"{'=' * 60}")
    if dry_run:
        print(f"找到 {renamed_count} 个文件匹配模式（未执行重命名）")
        print(f"提示: 设置 dry_run=False 来实际执行重命名")
    else:
        print(f"成功重命名 {renamed_count} 个文件")
    print(f"{'=' * 60}\n")
    
    return renamed_count


def process_image_folders(root_folder, dry_run=True):
    """
    自动识别并处理所有 image_数字 格式的文件夹
    
    参数:
        root_folder: 包含多个image_数字文件夹的根目录
        dry_run: 是否为干运行模式
    
    返回:
        处理的文件总数
    """
    # 正则表达式匹配 image_数字 格式的文件夹
    folder_pattern = re.compile(r'^image_(\d+)$')
    
    # 查找所有符合模式的文件夹
    image_folders = []
    
    if not os.path.exists(root_folder):
        print(f"错误: 根目录 {root_folder} 不存在")
        return 0
    
    # 列出根目录下所有项目
    items = os.listdir(root_folder)
    
    for item in items:
        item_path = os.path.join(root_folder, item)
        # 检查是否为文件夹且符合 image_数字 格式
        if os.path.isdir(item_path) and folder_pattern.match(item):
            image_folders.append(item_path)
    
    # 按文件夹名称排序
    image_folders.sort()
    
    if not image_folders:
        print(f"\n在 {root_folder} 中未找到符合 'image_数字' 格式的文件夹")
        return 0
    
    print(f"\n{'=' * 60}")
    print(f"找到 {len(image_folders)} 个 image_数字 文件夹:")
    for folder in image_folders:
        folder_name = os.path.basename(folder)
        file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        print(f"  • {folder_name} ({file_count} 个文件)")
    print(f"{'=' * 60}\n")
    
    # 处理每个文件夹
    total_renamed = 0
    
    for folder in image_folders:
        count = remove_frame_suffix(folder, dry_run)
        total_renamed += count
    
    print(f"\n{'#' * 60}")
    print(f"全部处理完成!")
    print(f"总计处理: {total_renamed} 个文件")
    print(f"{'#' * 60}\n")
    
    return total_renamed


def process_folder_tree(root_folder, dry_run=True):
    """
    递归处理文件夹树中所有子文件夹
    
    参数:
        root_folder: 根文件夹路径
        dry_run: 是否为干运行模式
    """
    total_renamed = 0
    
    # 遍历所有子文件夹
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if filenames:  # 如果文件夹中有文件
            count = remove_frame_suffix(dirpath, dry_run)
            total_renamed += count
    
    print(f"\n总计处理: {total_renamed} 个文件")
    return total_renamed


if __name__ == "__main__":
    # 设置包含多个 image_数字 文件夹的根目录
    root_directory = "/mnt/d/多帧扫描/1.2.0.20251014.153553.mice2-10帧/1/split_output/"
    
    print("删除文件后缀工具")
    print("=" * 60)
    print("此工具将去除文件名中的 '_frame_数字' 后缀")
    print("例如: filename_frame_1.raw → filename.raw")
    print("=" * 60)
    print(f"\n根目录: {root_directory}")
    print("将自动处理所有 image_数字 格式的子文件夹")
    
    # 先执行干运行，预览将要重命名的文件
    print("\n" + "=" * 60)
    print("第一步：预览模式")
    print("=" * 60)
    count = process_image_folders(root_directory, dry_run=True)
    
    if count > 0:
        user_input = input("\n是否执行重命名？(y/n): ").strip().lower()
        if user_input == 'y':
            print("\n" + "=" * 60)
            print("第二步：执行重命名")
            print("=" * 60)
            process_image_folders(root_directory, dry_run=False)
        else:
            print("\n已取消操作")
    else:
        print("\n未找到需要处理的文件")
    
    # 其他处理选项（已注释）
    """
    # 选项1: 处理单个文件夹
    target_folder = "/mnt/d/多帧扫描/1.2.0.20251014.144611.mice-10帧/1/image_5"
    print("\n处理单个文件夹模式")
    print(f"目标文件夹: {target_folder}")
    
    print("\n--- 第一步：预览 ---")
    count = remove_frame_suffix(target_folder, dry_run=True)
    
    if count > 0:
        user_input = input("\n是否执行重命名？(y/n): ").strip().lower()
        if user_input == 'y':
            print("\n--- 第二步：执行重命名 ---")
            remove_frame_suffix(target_folder, dry_run=False)
        else:
            print("\n已取消操作")
    
    # 选项2: 递归处理整个文件夹树
    print("\n\n递归处理整个文件夹树")
    print(f"根目录: {target_folder}")
    
    print("\n--- 第一步：预览 ---")
    count = process_folder_tree(target_folder, dry_run=True)
    
    if count > 0:
        user_input = input("\n是否执行重命名？(y/n): ").strip().lower()
        if user_input == 'y':
            print("\n--- 第二步：执行重命名 ---")
            process_folder_tree(target_folder, dry_run=False)
        else:
            print("\n已取消操作")
    """