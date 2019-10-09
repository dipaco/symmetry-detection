import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def gen_symmetry_fig(FLAGS, step, points, pred_normal, gt_plane):

    figs_path = os.path.join(FLAGS.log_dir, 'figs')
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    # --- Shows all the normal vectors in the batch vectors ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Plane normals in the entire batch(batch_size={})'.format(pred_normal.shape[0]))
    batch_plane_normals_fname = os.path.join(figs_path, 'step_{}_batch_plane_normals.png'.format(step))
    for i in range(pred_normal.shape[0]):
        _add_vector_arrow(ax, pred_normal[i, ...], color='red')
    _set_unit_limits_in_3d_plot(ax)
    plt.savefig(batch_plane_normals_fname)
    plt.close(fig)
    # ------

    # --- Shows the point cloud representing the object ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Shows the ground truth symmetry plane
    normal = gt_plane[0, ...]
    normal /= np.linalg.norm(normal)
    _add_plane(ax, normal, color='red', alpha=0.2)

    # Shows the estimated symmetry plane
    normal_est = pred_normal[0, ...]
    normal_est /= np.linalg.norm(normal_est)
    _add_plane(ax, normal_est, color='green', alpha=0.2)

    # Shows the cosine of the angle between planes
    cos_theta = np.sum(normal * normal_est)
    plt.rc('text', usetex=True)
    plt.title('Angle bt. ground truth plane and estimated plane \n' + r'$cos(\theta) = {:.3f}$'.format(cos_theta))
    plt.rc('text', usetex=False)

    point_cloud_fname = _show_point_cloud(ax, step, fig, points[0, ...], figs_path, '')
    plt.savefig(point_cloud_fname)
    plt.close(fig)
    # ------

    # Creates a list with all the filenames to log
    figures_filenames = [point_cloud_fname, batch_plane_normals_fname]

    return figures_filenames


def _add_vector_arrow(ax, axis, color='r', length=1.0, linewidth=0.5):
    axis_norm = length * axis / np.linalg.norm(axis)
    X, Y, Z = 0, 0, 0
    U, V, W = axis_norm[0], axis_norm[1], axis_norm[2]
    ax.quiver(X, Y, Z, U, V, W, color=color, linewidth=linewidth)


def _set_unit_limits_in_3d_plot(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_zticks([-1, 0, 1])
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])


def _show_point_cloud(ax, step, fig, points, figs_folder, name):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='.')
    _set_unit_limits_in_3d_plot(ax)
    point_cloud_fname = os.path.join(figs_folder, 'step_{}_{}_point_cloud.png'.format(step, name))
    return point_cloud_fname

def _add_plane(ax, normal, color='blue', alpha=0.1, show_normal=True):

    xx, yy = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    zz = np.zeros_like(xx)

    rot_mat, _, _ = axis_alignment_rotmat(normal)

    rot_plane = np.einsum('ij,jp->ip', rot_mat.T, np.vstack((xx.ravel(), yy.ravel(), zz.ravel()))).T

    xx = rot_plane[:, 0].reshape(xx.shape)
    yy = rot_plane[:, 1].reshape(yy.shape)
    zz = rot_plane[:, 2].reshape(zz.shape)

    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color)

    if show_normal:
        _add_vector_arrow(ax, normal, color=color, linewidth=1.0)


def _prepare_fig(ax, axis, name, triangles, title=''):

    #plt.title(name + '\n({:.2f}, {:.2f}, {:.2f})'.format(axis[0, 0], axis[0, 1], axis[0, 2]))
    plt.title(title)
    # ax.scatter(points[0, :, 0, 0], points[0, :, 1, 0], points[0, :, 2, 0], c='b', marker='.')
    # ax.plot_surface(points[0, :, 0, 0], points[0, :, 1, 0], points[0, :, 2, 0], c='b', marker='.')
    # ax.plot_trisurf(points[0, :, 0, 0], points[0, :, 1, 0], points[0, :, 2, 0])
    # ax.plot_trisurf(matplotlib.tri.Triangulation(points[0, :, 0, 0], points[0, :, 1, 0]))
    # ax.plot_wireframe(points[0, :, 0, 0], points[0, :, 1, 0], points[0, :, 2, 0], rstride=10, cstride=10)

    ax.add_collection3d(Poly3DCollection(triangles, facecolors='blue', linewidths=.1, edgecolors='black', alpha=0.1))
    _set_unit_limits_in_3d_plot(ax)


def axis_alignment_rotmat(central_axis):
    ''' Converts from axis angle to a rotation matrix
    Input:
    z - (3,) axis angle representation
    '''
    z = np.array([0, 0, 1])  # z axis
    eps = 1e-8

    central_axis_norm = np.linalg.norm(central_axis)
    normalized_axis = central_axis / central_axis_norm
    # print("normalized_axis{}, ||normalized_axis||={}".format(normalized_axis, np.linalg.norm(normalized_axis)))

    # infinitesimal rotation from normalized_axis
    central_axis_matrix = np.zeros((3, 3))
    # skew symmetric representation from normalized_axis
    central_axis_matrix[1, 0] = normalized_axis[2]
    central_axis_matrix[2, 0] = -normalized_axis[1]
    central_axis_matrix[2, 1] = normalized_axis[0]
    # print("central_axis_matrix={}".format(central_axis_matrix))
    central_axis_matrix = central_axis_matrix - central_axis_matrix.T
    # print("central_axis_matrix={}".format(central_axis_matrix))

    v = np.dot(central_axis_matrix, z)
    # print('test v is orthogonal', np.dot(v, z), np.dot(v, central_axis))

    omega = np.zeros((3, 3))
    # skew symmetric representation from v
    omega[1, 0] = v[2]
    omega[2, 0] = -v[1]
    omega[2, 1] = v[0]

    omega = omega - omega.T

    cos_theta = np.dot(normalized_axis, z)
    # print('cos theta={}'.format(cos_theta))

    # rodrigues formula
    R = np.eye(3) + omega + 1 / (1 + np.max([cos_theta, -1 + eps])) * np.dot(omega, omega)

    return R, cos_theta, omega