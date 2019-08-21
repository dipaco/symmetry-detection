import trimesh
import os


def load_off(filename):
    f = open(filename)
    return trimesh.load(f, 'off')


def generate_net_spec(indims, n_classes, n_layers, padding='VALID'):
    '''
    Generate network specification (i.e. filter_size, stride)
    Trys for uniform filter sizes with a smaller is better policy 
    First layer is constrained to have stride 1
    '''
    # symmetric
    up_len = int((n_layers-2)/2)
    filt_depth = [ 16 * 2**i for i in range(up_len) ]
    tmp = filt_depth[::-1]
    if n_layers%2:
        filt_depth.append( 16 * 2**up_len )
    filt_depth.extend( tmp )
    filt_depth.append( n_classes )
    filt_size = round((indims[1] - 1 + n_layers)/n_layers)
    
    return [(filt_size,f) for f in filt_out]


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

