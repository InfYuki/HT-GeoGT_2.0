'''
====================================================
    PathFinder 重写版本 - GPU优化
====================================================
'''
import random
import torch


def get_halfpath(graph, start_node, neighbor, max_hop):
    """获取一条从起始节点到某个端点的路径（halfpath）- GPU优化版本

    参数:
    graph: 图字典
    start_node: 起始节点
    neighbor: 第一跳邻居节点
    max_hop: 最大跳数

    返回:
    list: 路径节点列表，长度不足的部分用-1填充
    """
    visited = {start_node, neighbor}
    path = [start_node, neighbor]

    current = neighbor
    hop_count = 1

    while hop_count < max_hop:
        # 获取邻居节点
        edge_index = graph['edge_index']
        neighbors = []

        # 找出所有与当前节点相连的边
        for i in range(edge_index.shape[1]):
            if edge_index[0, i].item() == current:
                next_node = edge_index[1, i].item()
                if next_node not in visited:
                    neighbors.append(next_node)

        if not neighbors:
            break

        next_node = random.choice(neighbors)
        path.append(next_node)
        visited.add(next_node)
        current = next_node
        hop_count += 1

    # 用-1填充剩余长度
    while len(path) < max_hop + 1:
        path.append(-1)

    return path


def combine_halfpaths(path1, path2):
    """将两条halfpath组合成一条centerpath

    参数:
    path1, path2: 两条halfpath

    返回:
    list: 组合后的centerpath
    """
    # 计算非填充部分的长度
    non_fill_len1 = path1.index(-1) if -1 in path1 else len(path1)
    non_fill_len2 = path2.index(-1) if -1 in path2 else len(path2)

    # 复制路径以避免修改原始路径
    path1_copy = path1.copy()
    path2_copy = path2.copy()

    # 找到较长的非填充部分
    if non_fill_len1 > non_fill_len2:
        # 用-1填充path1多余的非填充部分
        for i in range(non_fill_len2, non_fill_len1):
            path1_copy[i] = -1
    elif non_fill_len2 > non_fill_len1:
        # 用-1填充path2多余的非填充部分
        for i in range(non_fill_len1, non_fill_len2):
            path2_copy[i] = -1

    # 反转第一条路径（除了中心节点），然后拼接第二条路径
    reversed_path1 = path1_copy[::-1]
    centerpath = reversed_path1[:-1] + path2_copy

    return centerpath


def FindCenterPath_Node(max_num_path, max_num_hop, node, graph, k=2):
    """生成以一个节点为中心的两端等跳数长的数条路径 - GPU优化版本

    参数:
    max_num_path: 最大CenterPath条数
    max_num_hop: CenterPath的最大长度
    node: 目标节点
    graph: 该节点所在的图
    k: 最小度数阈值，默认为2

    返回:
    dict: 包含节点信息和该节点上所有的centerpath
    """
    # 计算节点的度
    edge_index = graph['edge_index']
    node_neighbors = []

    # 找出所有与该节点相连的边 - 使用GPU加速
    mask = (edge_index[0] == node)
    if torch.any(mask):
        neighbors = edge_index[1, mask].tolist()
        node_neighbors = list(set(neighbors))  # 去重

    node_d = len(node_neighbors)

    result = {
        'node': node,
        'centerpathlist': []
    }

    if node_d >= k:  # 使用参数k作为度数阈值
        # 为每个邻居生成一条halfpath
        halfpaths = []
        used_neighbors = set()

        for neighbor in node_neighbors:
            # 确保不使用重复的第一跳邻居
            if neighbor in used_neighbors:
                continue

            used_neighbors.add(neighbor)

            if len(halfpaths) >= max_num_path:
                break

            halfpath = get_halfpath(graph, node, neighbor, max_num_hop)
            if halfpath:
                halfpaths.append(halfpath)

        halfpath_num = len(halfpaths)

        if halfpath_num >= 2:
            # 如果halfpath数量为奇数，随机复制一条
            if halfpath_num % 2 == 1:
                random_idx = random.randint(0, halfpath_num - 1)
                halfpaths.append(halfpaths[random_idx].copy())
                halfpath_num += 1

            # 两两配对组合成centerpath
            for i in range(0, halfpath_num, 2):
                if i + 1 < halfpath_num:
                    centerpath = combine_halfpaths(halfpaths[i], halfpaths[i + 1])
                    result['centerpathlist'].append(centerpath)

    return result