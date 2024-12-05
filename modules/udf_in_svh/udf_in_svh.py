import numpy as np
import torch
import open3d as o3d

torch.classes.load_library("sparse_hash/libsvh.so")

# num_samples为选择voxel的个数
def uniform_sample_svh(svh_dic, num_samples, device = "cuda"):
    vox_coords = svh_dic["ht_info"][:, :3]
    inval_val = svh_dic["inval_val"]    #无效坐标值，与c++类中对应
    vox_coords = vox_coords[vox_coords[:, 0] != inval_val].to(device)
    voxel_size = svh_dic["voxel_size"]
    pts = svh_dic["lcp_array"]    # 点云
    vox_coords = (vox_coords+0.5) * voxel_size    #得到体素中心坐标
    num_voxels = vox_coords.shape[0]     # voxel的个数
    num_pts = pts.shape[0]                # 点云中点的个数
    
    idx = np.random.randint(0, num_pts, num_samples//3)
    pts = pts[idx].to(device)
    
    # 得到在哈希体素中随机采样点
    idx = np.random.randint(0, num_voxels, num_samples)    # 随机选择num_samples个voxel
    vox_coords = vox_coords[idx]
    num_per_voxel = 2
    x_vox = torch.empty([num_samples, num_per_voxel, 3], device=device).uniform_(-0.5,0.5)
    x_vox *= voxel_size
    x_w = x_vox + vox_coords.unsqueeze(1)
    x_w = x_w.view(-1, 3)

    # x_w = torch.cat([pts, x_w], dim = 0)
    
    return x_w 
    
def compute_pointcloud_dis_svh(svh, ht_info, lcp_index, lcp_array, pts, device = "cuda"):   #pts为输入的点云[N, 3]，输出dis[N]
    G = 256
    N = pts.shape[0]  #N为光线数
    K = int(np.ceil(N / G))
    H = K * G
    if H > N:
        pts = torch.cat([pts, pts[: H - N]], 0)   
    pts = pts.reshape(G, K, 3).contiguous()     #需要将数组变为连续的
    ht_info = ht_info.expand(G, *ht_info.size()).contiguous()
    index = svh.find_from_ht_neighborhood(pts, ht_info)   #查找点所处的哈希voxel的索引（6邻域查找）
    del ht_info   #释放空间

    index = index.reshape(H, )
    index_ = torch.unique(index, sorted = True).long()    #取唯一值，得到与输入点云相交的哈希voxel索引

    #由pts得到pt_lcp_index，再得到lcp_index，最后得到每个pt指向所在的局部点云lcp_array
    temp = torch.arange(len(index_)).to(device)
    pt_lcp_index = (torch.eq(index[None, :], index_[:, None])*temp[:, None]).sum(dim = 0).int()
    pt_lcp_index = pt_lcp_index.reshape(G, K).contiguous() 

    lcp_index = lcp_index[index_].to("cpu")    #得到相交哈希voxel中局部点云在点云数组中的索引
    lcp_ids =  torch.cat([torch.arange(cp_id, cp_id+cp_len) for cp_id, cp_len in lcp_index])   #这种形式的for循环和cat一块使用，加速有5倍

    lcp_ids = lcp_ids.long()                     #得到局部点云在点云数组中的索引
    lcp_array = lcp_array[lcp_ids]       #得到局部点云，其每个voxel的起始索引和点云长度保存在lcp_index中

    #得到相交哈希voxel中局部点云在点云数组中重新排列后的索引
    lcp_index[:, 0] = torch.cumsum(torch.cat([torch.tensor([0]), lcp_index[:, 1][:-1]]), dim = 0)
    lcp_index = lcp_index.to(device)
    lcp_array = lcp_array.to(device)
    lcp_index = lcp_index.expand(G, *lcp_index.size()).contiguous()
    lcp_array = lcp_array.expand(G, *lcp_array.size()).contiguous()

    dis = svh.compute_pointcloud_dis(pts, pt_lcp_index, lcp_index, lcp_array)

    dis = dis.reshape(H, )
    if H > N:
        dis = dis[:N]
        index = index[:N]
    valid_mask = index>=0     #为有效距离掩膜
        
    return dis, valid_mask

if __name__ == "__main__":
    device = "cuda"
    # 加载点云
    pcd = o3d.io.read_point_cloud("./pointcloud.ply")
    points = np.asarray(pcd.points)
    points = torch.from_numpy(points).float()
    
    # 初始化svh，构建哈希网格，并向其中插入局部点云
    voxel_size = 1.5
    ht_size = 0x10000
    svh = torch.classes.svh.HashTable(voxel_size, ht_size)    #初始化哈希表
    svh.insert(points)     #哈希表插入操作
    svh.insert_local_pointcloud(points)  #向哈希表中插入局部点云
    ht_info, lcp_index, lcp_array = svh.get_ht_info()
    svh_dic = {"ht_info": ht_info, "lcp_index": lcp_index, "lcp_array": lcp_array, "voxel_size": voxel_size, "ht_size": ht_size, "inval_val": 999999}
    
    pts = uniform_sample_svh(svh_dic, num_samples = 2**15)   # 在哈希网格中均匀采样
    dis, valid_mask = compute_pointcloud_dis_svh(svh, ht_info.to(device), lcp_index.to(device), lcp_array.to(device), pts.to(device))   # 并行计算每个采样点到场景点云的最近距离，作为udf值
    
    pts = pts[valid_mask].cpu().numpy()
    dis = dis[valid_mask].cpu().numpy()
    max_dis = 1.1732*voxel_size
    dis_normalize = dis / max_dis
    colors = np.ones_like(pts)
    colors[:,0] -= dis_normalize
    colors[:,1] -= dis_normalize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)    # 用udf值进行着色，值越大颜色越深
    
    vox_coord = (ht_info[:, :3]) * voxel_size
    inval_val = 999999
    vox_coord = vox_coord[torch.where(ht_info[:, 0]!=inval_val)].numpy()
    # 画网格线
    offset = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])*voxel_size
    connect = np.array([[0,1], [0,2], [2,3], [1,3], [0,4], [1,5], [2,6], [3,7], [4,5], [4,6], [5,7], [6, 7]])
    pts = np.concatenate([(coord[None, :] + offset) for coord in vox_coord], axis = 0)
    lines = np.concatenate([(connect + i*8) for i in range(len(vox_coord))], axis = 0)
    line_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(pts),
        lines = o3d.utility.Vector2iVector(lines)
    )
    colors = np.array([[0.6, 0.0, 0.0] for i in range(len(lines))])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # 采样点与哈希网格可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window()	
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	
    render_option.background_color = np.array([0,0,0])	
    render_option.point_size = 1.5	
    render_option.line_width = 0.3 #1  
    vis.add_geometry(pcd)	#添加点云
    vis.add_geometry(line_set)	#添加线
    vis.run()
    
    
    
   