import multiprocessing

from enum import Enum

class Device(Enum):
    CPU = 0
    CUDA = 1


class config:

    # General options
    netname = None # the network name, the extension can be only .pyt, .tf and .meta
    epsilon = 0 # the epsilon for L_infinity perturbation
    zonotope = None # file to specify the zonotope matrix
    domain = None # the domain name can be either deepzono, refinezono, deeppoly or refinepoly
    dataset = None # the dataset, can be either mnist, cifar10, or acasxu
    complete = False # flag specifying where to use complete verification or not
    timeout_lp = 1 # timeout for the LP solver
    timeout_milp = 1 # timeout for the MILP solver
    use_default_heuristic = True # whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation
    mean = None # the mean used to normalize the data with
    std = None # the standard deviation used to normalize the data with
    num_tests = None # Number of images to test
    from_test = 0 # From which number to start testing
    debug = False # Whether to display debug info
    subset = None
    target = None # 
    epsfile = None

    # refine options
    use_milp = True # Whether to use MILP
    refine_neurons = False # refine neurons
    sparse_n = 70
    numproc = multiprocessing.cpu_count() # number of processes for milp/lp/krelu
    normalized_region = True
    # Geometric options
    geometric = False # Whether to do geometric analysis
    attack = False # Whether to attack in geometric analysis
    data_dir = None # data location for geometric analysis
    geometric_config = None # geometric config location
    num_params = 0 # Number of transformation parameters for geometric analysis

    # Acas Xu
    specnumber = None # Acas Xu spec number

    # arbitrary input / output
    input_box = None # input box file to use
    output_constraints = None # output constraints file to check

    # GPU options
    device = Device.CPU # Which device Deeppoly should run on
