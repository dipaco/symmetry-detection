import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pylab as plt
import numpy as np
import os


def rot_to_trans(R):
    T = np.zeros((4,4))
    T[:3, :3] = R
    T[-1,-1] = 1
    
    return T


def get_rotation(angles):
    alpha, beta, gamma = angles
    R = np.dot(np.dot(rot_z(alpha), rot_y(beta)), rot_z(gamma))
    return R


def rot_z(theta):
    return np.array([ [np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1] ])


def rot_y(theta):
    return np.array([ [np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)] ])


def is_convex_combination(p, tri):
    '''
    Determines if point p is contained in tri by checking if 
    it can be described as the convex combination of the points
    that determine tri

    Input:
    p - (n,) the point being checked
    tri - (n,3) the triangle to be contained in
    n \in [2, 3]
    '''
    n = p.shape[0]

    if n == 2:
        tri = np.vstack((tri, np.ones((1,3))))
        p = np.vstack((p, 1))

    elif n != 3:
        raise ValueError('Allowable input point dimensions are 2 or 3')
        
    x = np.linalg.lstsq(tri, p)[0]
    # print("tri = {}, p = {}, x = {}, pos(x) = {}, sum(x) = {}".format(tri, p, x, np.all(x>0), np.sum(x)))

    return np.all(x > 0) and np.isclose(np.sum(x), 1)


def axis_angle_to_rotmat(z):
    ''' Converts from axis angle to a rotation matrix 
    Input:
    z - (3,) axis angle representation
    '''

    # theta=angle, z_n = unit norm direction
    theta = np.linalg.norm(z)
    z_n = z/theta
    # print("theta={}, ||z_n||={}".format(theta, np.linalg.norm(z_n)))

    # infinitesimal rotation from z_n
    omega = np.zeros((3,3))
    # skew symmetric representation from z_n
    omega[1,0] = z_n[2]
    omega[2,0] = -z_n[1]
    omega[2,1] = z_n[0]
    # print("omega={}".format(omega))
    omega = omega - omega.T
    # print("omega={}".format(omega))

    # rodrigues formula
    R = np.eye(3) + np.sin(theta) * omega + (1 - np.cos(theta)) * np.dot(omega, omega)

    return R, theta, z_n


def axis_alignment_rotmat(central_axis):
    ''' Converts from axis angle to a rotation matrix 
    Input:
    z - (3,) axis angle representation
    '''
    z = np.array([0, 0, 1]) # z axis
    eps = 1e-8

    central_axis_norm = np.linalg.norm(central_axis)
    normalized_axis = central_axis/central_axis_norm
    # print("normalized_axis{}, ||normalized_axis||={}".format(normalized_axis, np.linalg.norm(normalized_axis)))

    # infinitesimal rotation from normalized_axis
    central_axis_matrix = np.zeros((3,3))
    # skew symmetric representation from normalized_axis
    central_axis_matrix[1,0] = normalized_axis[2]
    central_axis_matrix[2,0] = -normalized_axis[1]
    central_axis_matrix[2,1] = normalized_axis[0]
    # print("central_axis_matrix={}".format(central_axis_matrix))
    central_axis_matrix = central_axis_matrix - central_axis_matrix.T
    # print("central_axis_matrix={}".format(central_axis_matrix))

    v = np.dot( central_axis_matrix, z )
    # print('test v is orthogonal', np.dot(v, z), np.dot(v, central_axis))

    omega = np.zeros((3,3))
    # skew symmetric representation from v
    omega[1,0] = v[2]
    omega[2,0] = -v[1]
    omega[2,1] = v[0]

    omega = omega - omega.T
    
    cos_theta = np.dot(normalized_axis, z)
    # print('cos theta={}'.format(cos_theta))
    
    # rodrigues formula
    R = np.eye(3) + omega + 1/(1 + np.max([cos_theta, -1 + eps])) * np.dot(omega, omega)

    return R, cos_theta, omega


def spherical_to_cartesian(phi, theta):
    ''' As in wiki
    '''

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.array([x, y, z])


def gen_supervised_symmetry_fig(FLAGS, step, points, triangles, est_plane_normal, name, symmetry_plane):

    figs_path = os.path.join(FLAGS.checkpoint_save_dir, 'figs')
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    # Shows all the normal vectors in the batch vectors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Plane normals in the entire batch(batch_size={})'.format(est_plane_normal.shape[0]))
    batch_plane_normals_fname = os.path.join(figs_path, 'step_{}_{}_batch_plane_normals.png'.format(step, name))
    for i in range(est_plane_normal.shape[0]):
        _add_vector_arrow(ax, est_plane_normal[i, ...], color='red')
    _set_unit_limits_in_3d_plot(ax)
    plt.savefig(batch_plane_normals_fname)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    point_cloud_fname = _show_point_cloud(ax, step, fig, points, figs_path, name[0, 0].decode("utf-8"))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Shows the mesh
    name = name[0, 0].decode("utf-8")
    _prepare_fig(ax, est_plane_normal[-1, ...], name, np.transpose(triangles[0, ...], axes=[0, 2, 1]))

    # Shows the ground truth symmetry plane
    normal = symmetry_plane[0, ...]
    normal /= np.linalg.norm(normal)
    _add_plane(ax, normal, color='red', alpha=0.2)

    # Shows the estimated symmetry plane
    normal_est = est_plane_normal[0, ...]
    normal_est /= np.linalg.norm(normal_est)
    _add_plane(ax, normal_est, color='green', alpha=0.2)

    # Shows the cosine of the angle between planes
    cos_theta = np.sum(normal * normal_est)
    plt.rc('text', usetex=True)
    plt.title('Angle bt. ground truth plane and estimated plane \n' + r'$cos(\theta) = {:.3f}$'.format(cos_theta))
    plt.rc('text', usetex=False)

    est_plane_fname = os.path.join(figs_path, 'step_{}_{}_est_plane_and_gt.png'.format(step, name))
    plt.savefig(est_plane_fname)
    plt.close(fig)

    figures_filenames = []
    figures_filenames.append(est_plane_fname)
    figures_filenames.append(point_cloud_fname)
    figures_filenames.append(batch_plane_normals_fname)

    return figures_filenames

def gen_symmetry_fig(FLAGS, step, points, triangles, est_plane_normal, name, rot_matrix, cyl_proj, flipped_cyl_proj, symmetry_plane, cyl_axis):

    figs_path = os.path.join(FLAGS.checkpoint_save_dir, 'figs')
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    # Shows all the normal vectors in the batch vectors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Plane normals in the entire batch(batch_size={})'.format(est_plane_normal.shape[0]))
    batch_plane_normals_fname = os.path.join(figs_path, 'step_{}_{}_batch_plane_normals.png'.format(step, name))
    for i in range(est_plane_normal.shape[0]):
        _add_vector_arrow(ax, est_plane_normal[i, ...], color='red')
    _set_unit_limits_in_3d_plot(ax)
    plt.savefig(batch_plane_normals_fname)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    point_cloud_fname = _show_point_cloud(ax, step, fig, points, figs_path, name[0, 0].decode("utf-8"))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Shows the mesh
    name = name[0, 0].decode("utf-8")
    _prepare_fig(ax, est_plane_normal[-1, ...], name, np.transpose(triangles[0, ...], axes=[0, 2, 1]))

    # Shows all the axis vectors
    for i in range(cyl_axis.shape[1]):
        _add_vector_arrow(ax, cyl_axis[0, i, ...], color='black')

    # Shows the ground truth symmetry plane
    normal = symmetry_plane[0, ...]
    normal /= np.linalg.norm(normal)
    _add_plane(ax, normal, color='red', alpha=0.2)

    # Shows the estimated symmetry plane
    normal_est = est_plane_normal[0, ...]
    normal_est /= np.linalg.norm(normal_est)
    _add_plane(ax, normal_est, color='green', alpha=0.2)

    # Shows the cosine of the angle between planes
    cos_theta = np.sum(normal * normal_est)
    plt.rc('text', usetex=True)
    plt.title('Angle bt. ground truth plane and estimated plane \n' + r'$cos(\theta) = {:.3f}$'.format(cos_theta))
    plt.rc('text', usetex=False)

    est_plane_fname = os.path.join(figs_path, 'step_{}_{}_est_plane_and_gt.png'.format(step, name))
    plt.savefig(est_plane_fname)
    plt.close(fig)

    figures_filenames = []
    for i in range(cyl_proj.shape[1]):
        #plt.rc('text', usetex=True)

        f = cyl_proj[0, i, ..., 0]
        g = flipped_cyl_proj[0, i, ..., 0]
        corr = np.sum(f*g)

        #ax = fig.add_subplot(111)
        #plt.title(r"Correlation between f and its flipped version g. \n $\gamma = {}$".format(np.round(corr, 2)))

        fig = plt.figure()

        plt.subplot(121)
        plt.title(r"Correlation $\gamma = {:0.2f}$".format(corr))
        plt.imshow(f)
        plt.subplot(122)
        plt.imshow(g)
        figures_filenames.append(os.path.join(figs_path, 'step_{}_{}_cyl_proj_{}.png'.format(step, name, i)))
        plt.savefig(figures_filenames[i])
        plt.close(fig)

    figures_filenames.append(est_plane_fname)
    figures_filenames.append(point_cloud_fname)
    figures_filenames.append(batch_plane_normals_fname)

    return figures_filenames


def _show_point_cloud(ax, step, fig, points, figs_folder, name):
    ax.scatter(points[0, :, 0], points[0, :, 1], points[0, :, 2], c='blue', marker='.')

    _set_unit_limits_in_3d_plot(ax)

    point_cloud_fname = os.path.join(figs_folder, 'step_{}_{}_point_cloud.png'.format(step, name))
    plt.savefig(point_cloud_fname)
    plt.close(fig)

    return point_cloud_fname


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


def _add_vector_arrow(ax, axis, color='r', length=1.0, linewidth=0.5):
    axis_norm = length * axis / np.linalg.norm(axis)
    X, Y, Z = 0, 0, 0
    U, V, W = axis_norm[0], axis_norm[1], axis_norm[2]
    ax.quiver(X, Y, Z, U, V, W, color=color, linewidth=linewidth)


def gen_fig(FLAGS, points, triangles, axis, name, rot_matrix, cyl_proj):

    figs_path = os.path.join(FLAGS.checkpoint_save_dir, 'figs')
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _prepare_fig(ax, axis, name[0, 0].decode("utf-8"), np.transpose(triangles[0, ...], axes=[0, 2, 1]))
    _add_vector_arrow(ax, axis[0, :], length=1.5)

    init_pose_f_name = os.path.join(figs_path, '{}_init_pose.png'.format(name[0, 0].decode("utf-8")))
    plt.savefig(init_pose_f_name)
    plt.close(fig)

    rot_triangles = np.einsum('bij,bfjp->bfip', rot_matrix, triangles)
    rot_axis = np.einsum('bij,bj->bi', rot_matrix, axis)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _prepare_fig(ax, rot_axis, name[0, 0].decode("utf-8"), np.transpose(rot_triangles[0, ...], axes=[0, 2, 1]))
    _add_vector_arrow(ax, axis[0, :], length=1.5)

    # Cylinder
    x = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 100)
    Xc, Zc = np.meshgrid(x, z)
    Yc = np.sqrt(1 - Xc ** 2)

    # Draw parameters
    rstride = 20
    cstride = 10
    ax.plot_surface(Xc, Yc, Zc, alpha=0.1, rstride=rstride, cstride=cstride, color='g')
    ax.plot_surface(Xc, -Yc, Zc, alpha=0.1, rstride=rstride, cstride=cstride, color='g')


    rot_pose_fname = os.path.join(figs_path, '{}_rot_pose.png'.format(name[0, 0].decode("utf-8")))
    plt.savefig(rot_pose_fname)
    plt.close(fig)

    fig.add_subplot(133)
    plt.imshow(cyl_proj[0, ..., 0])
    cyl_proj_fname = os.path.join(figs_path, '{}_cyl_proj.png'.format(name[0, 0].decode("utf-8")))
    plt.savefig(cyl_proj_fname)
    plt.close(fig)

    return init_pose_f_name, rot_pose_fname, cyl_proj_fname


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

def align_mesh(query, obj):

    max_i = 0
    max_prod = 0
    for i in range(1, obj['cyl_proj'].shape[1]):
        prod = (np.roll(obj['cyl_proj'], shift=i, axis=1) * query['cyl_proj']).sum()

        if prod > max_prod:
            max_prod = prod
            max_i = i

    print(max_i)

    az_angle = - max_i * (2*np.pi) / query['cyl_proj'].shape[1]

    az_rot_mat = np.array([[[np.cos(az_angle), -np.sin(az_angle), 0.0],
                            [np.sin(az_angle), np.cos(az_angle), 0.0],
                            [0.0, 0.0, 1.0]]])
    total_rot_mat = np.einsum('bij,bjp->bip', az_rot_mat, obj['rot_matrix'])
    total_rot_mat = np.einsum('bij,bjp->bip', np.transpose(query['rot_matrix'], axes=[0, 2, 1]), total_rot_mat)

    #rot_obj_triangles = np.einsum('bij,bfjp->bfip', obj['rot_matrix'], obj['triangles'])

    aligned_triangles = np.einsum('bij,bfjp->bfip', total_rot_mat, obj['triangles'])

    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    _prepare_fig(ax, np.array([[0.000, 0.000, 0.000]]), 'query', query['triangles'])

    ax = fig.add_subplot(132, projection='3d')
    _prepare_fig(ax, np.array([[0.000, 0.000, 0.000]]), 'unaligned', obj['triangles'])

    ax = fig.add_subplot(133, projection='3d')
    _prepare_fig(ax, np.array([[0.000, 0.000, 0.000]]), 'aligned', aligned_triangles)

    init_pose_f_name = os.path.join('Users/dipaco/Trash/tests', '{}_aligned.png'.format('example'))
    plt.show()
    #plt.savefig(init_pose_f_name)
    #plt.close(fig)

    aligned = {
                'triangles': aligned_triangles,
                'rot_matrix': None,
                'cyl_proj': None
              }
    return aligned


def cosine_similarity(estimated_planes, ground_truth_planes):
    pass