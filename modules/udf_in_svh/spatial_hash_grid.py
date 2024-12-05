import numpy as np
import torch
import open3d as o3d

torch.classes.load_library("sparse_hash/libsvh.so")

if __name__ == "__main__":
    
    # 加载点云
    pcd = o3d.io.read_point_cloud("./pointcloud.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    points = torch.from_numpy(points).float()
    colors = torch.from_numpy(colors).float()
    
    # 初始化svh，构建哈希网格，并向其中插入局部点云
    voxel_size = 1.0
    ht_size = 0x10000
    svh = torch.classes.svh.HashTable(voxel_size, ht_size)    #初始化哈希表
    svh.insert(points)     #哈希表插入操作
    svh.insert_local_pointcloud(points)  #向哈希表中插入局部点云
    ht_info, _, _ = svh.get_ht_info()
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
    colors = np.array([[190/255.0, 196/255.0, 218/255.0] for i in range(len(lines))])
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # 点云与哈希网格可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window()	
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	
    render_option.background_color = np.array([1, 1, 1])	
    render_option.point_size = 2	
    render_option.line_width = 1  
    vis.add_geometry(pcd)	#添加点云
    vis.add_geometry(line_set)	#添加线
    vis.run()