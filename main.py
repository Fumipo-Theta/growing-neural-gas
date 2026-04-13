import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from neuralgas import GrowingNeuralGas


def main(pointcloud_path: Path) -> None:
    pcd = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pcd.points)

    max_neurons = 5000
    max_iter = 5000
    max_age = 50
    eb = 0.05
    en = 0.005
    alpha = 0.5
    beta = 0.0005
    l = 100

    gng = GrowingNeuralGas(
        max_neurons=max_neurons,
        max_iter=max_iter,
        max_age=max_age,
        eb=eb,
        en=en,
        alpha=alpha,
        beta=beta,
        l=l,
        dataset=points)
    gng.learn()

    graph = gng.gng
    nodes = graph.vs
    edges = graph.es

    print(f"Number of neurons: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")

    nodes_mesh = o3d.geometry.TriangleMesh()
    node_points = [node['weight'] for node in nodes]
    edge_indices = [(edge.source, edge.target) for edge in edges]

    for node in node_points:
        nodes_mesh += o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(node)
    nodes_mesh.paint_uniform_color([0, 1, 0])

    edge_lines = o3d.geometry.LineSet()
    edge_lines.points = o3d.utility.Vector3dVector(node_points)
    edge_lines.lines = o3d.utility.Vector2iVector(edge_indices)
    edge_lines.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd, nodes_mesh, edge_lines])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Growing Neural Gas for 3D Point Cloud")
    parser.add_argument("pointcloud_path", type=Path, help="Path to the input point cloud file")
    args = parser.parse_args()

    main(**args.__dict__)
