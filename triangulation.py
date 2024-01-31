import numpy as np 
import scipy
from collections import defaultdict



class Hyperplane:
    """
    Forms a hyperplane from a set of points
    Assumes that 
    """
    def __init__(self, points, perform_checks=True):
        """
        D points in D dim space defines a hyperplane (assuming not colinear)
        shape(points) = (D,D)
        """
        self.D = points.shape[0]
        self.points = points
        self.perform_checks = perform_checks
        self.compute_normal_vector()

    def compute_normal_vector(self):
        """
        diffs[i] = points[i] - points[0] for i > 0
        shape(diffs) = (D,D-1)
        Normal vector to plane corresponds to the null space of 'diffs'
        So we can compute it using the SVD of 'diffs' and taking the unit vector corresponding to   
            the singular value of 0. (N.B. garunteed to have a singular value of 0, because shape(diffs)=(D,D-1))
        """
        diffs = np.zeros((self.D,self.D-1))
        for i in range(1,self.D):
            diffs[:,i-1] = self.points[:,i] - self.points[:,0]
        
        U, _S, _Vh = np.linalg.svd(diffs, full_matrices=True)
        self.normal = U[:,self.D-1]

        if self.perform_checks:
            for i in range(self.D-1):
                if not np.isclose(0.0, np.dot(self.normal, diffs[:,i])):
                    raise "Normal doesnt appeart to actually be normal to plane"
    
    def halfplane_test(self, p1, p2):
        """
        Checks if a points p1 and p2 are the same side of this hyperplane
        shape(p1) = (D,)
        shape(p2) = (D,)
        """
        delta1 = p1 - self.points[:,0]
        delta2 = p2 - self.points[:,0]

        dot1 = np.dot(delta1, self.normal)
        dot2 = np.dot(delta2, self.normal)

        # edge case where one of points is (almost) on plane
        if (np.isclose(0.0,dot1) or np.isclose(0.0,dot2)):
            return True
        
        # if same sign, then on same side
        return np.sign(dot1) * np.sign(dot2) > 0

    def point_is_normal_side(self, p):
        """
        Checks if p is on the side of the hyperplane that the normal points in
        shape(p) = (D,)
        """
        return np.dot(p - self.points[:,0], self.normal) >= 0

    def point_in_plane(self, p):
        return np.dot(p - self.points[:,0], self.normal) == 0
        # return np.isclose(0.0, np.dot(p - self.points[:,0], self.normal))



class Simplex:
    """
    Class for D-1-simplex in D dim space

    Let D be the dimension of space
    D points in simplex
    Simplex is a D-1-simplex

    For example, if D=3, then there are 3 points in the 2-simplex (i.e. a triangle)
    
    Note that the D points in the simplex also lies in a D-1 hyperplane
    - e.g. the 2-simplex (triangle) is a 2d subspace of 3d space
    """
    def __init__(self, points, perform_checks=True):
        """
        shape(points) = (D,D)
        """
        self.D = points.shape[0]
        if perform_checks and self.D < 3:
            raise "Simplex logic only works for 3+ dimensions, 2d case is just a line anyway :)"
        self.points = points
        self.simplex_hyperplane = Hyperplane(points,perform_checks)
        self.normal = self.simplex_hyperplane.normal
        self.perform_checks = perform_checks
        self.compute_edge_points()
    
    def compute_edge_points(self):
        """
        Computes 0.5*(u+v) for each u,v in the simplex
        """
        num_edge_points = int(0.5 * self.D * (self.D - 1))
        self.edge_points = np.zeros((self.D, num_edge_points))
        self.edge_point_to_edge = {}
        self.vertex_to_edge_points = defaultdict(list)
        i = 0
        for j in range(self.D):
            for k in range(j+1, self.D):
                ratio = 0.3 + 0.4 * np.random.rand()
                self.edge_points[:,i] = ratio * self.points[:,j] + (1.0-ratio) * self.points[:,k]
                self.edge_point_to_edge[tuple(self.edge_points[:,i])] = (j,k)
                self.vertex_to_edge_points[j].append(self.edge_points[:,i])
                self.vertex_to_edge_points[k].append(self.edge_points[:,i])
                i += 1

    def compute_max_l1_norm_ratio(self, base_max_l1_norm=2.0):
        """
        Computes the ratio of the maximum l1 norm of this simplex, with respect to some base_max_l1_norm
        For the unit simplex, the maximum l1 norm is 2.0
        When subdividing a simplex 'parent', into this simplex 'child', then:
            child.max_l1_norm = max_l1_norm_ratio * parent.max_l1_norm
        Default value of 2.0 is because we're only going to split the unit simplex in this script
        """
        max_l1_norm = 0.0
        for j in range(self.D):
            for k in range(j+1,self.D):
                l1_norm = np.norm(self.points[:,j]-self.points[:,k], ord=1)
                if l1_norm > max_l1_norm:
                    max_l1_norm = l1_norm
        self.max_l1_norm_ratio = max_l1_norm / base_max_l1_norm
        return self.max_l1_norm_ratio

    def compute_triangulation(self):
        self.triangulation = []

        # simplexes formed around vertices
        for i in range(self.D):
            simplex_points = np.zeros((self.D,self.D))
            for j, edge_pnt in enumerate(self.vertex_to_edge_points[i]):
                simplex_points[:,j] = edge_pnt
            simplex_points[:,self.D-1] = self.points[:,i]
            self.triangulation.append(Simplex(simplex_points,self.perform_checks))
        
        # Triangulation of internal space
        self.triangulation.extend(triangulation(self.edge_points, self.normal, self.perform_checks))
            
                



class UnitSimplex(Simplex):
    """
    Helper to make unit simplex
    The identity matrix in D dimensions forms a D-1-simplex
    """
    def __init__(self, D):
        super().__init__(np.identity(D))



def t_split_points(points, hyperplane_indices, normal):
    # psuedo point (because really need D points for a D-1 hyperplane), and simplex itself is in a hyperplane
    psuedo_point = np.expand_dims(points[:,0] + normal, axis=1)
    hyperplane_points = np.concatenate([points[:,hyperplane_indices], psuedo_point], axis=1)
    hyperplane = Hyperplane(hyperplane_points)

    num_points = points.shape[1]
    normal_side_indices = []
    opp_side_indices = []
    for j in range(num_points):
        if j in hyperplane_indices:
            continue
        if hyperplane.point_is_normal_side(points[:,j]):
            normal_side_indices.append(j)
        else:
            opp_side_indices.append(j)
    
    return normal_side_indices, opp_side_indices

def t(points, simplex_list, D, normal, perform_checks=True):
    """
    Working in the D-1 dimensional hyperplane that the simplex lies in
    Recursively finds a set of D-1 points, that define a D-2 dimesional hyperplane that divides the point
    - really we find a D-1 dimensional hyperplane that splits the points
    - this D-1 plane is the D-2 plane, with a psuedo point added, using the normal vector to the simplex plane
    """
    print("Called t with points shape={s}".format(s=points.shape))
    num_points = points.shape[1]

    # base case, we're a simplex already
    if num_points == D:
        simplex_list.append(Simplex(points))
        return simplex_list
    
    # try a reasonable number of hyperplanes that will find a split
    hyperplane_indices_to_try = []
    if num_points == D+1:
        for i in range(num_points):
            for j in range(i+1,num_points):
                hyperplane_indices = list(range(i)) + list(range(i+1,j)) + list(range(j+1,num_points))
                hyperplane_indices_to_try.append(hyperplane_indices)
    else:
        for i in range(D-2,num_points):
            hyperplane_indices = list(range(D-2)) + [i]
            hyperplane_indices_to_try.append(hyperplane_indices)

    # iterate through the hyperplanes to find the point that gives the largest split
    # really we just want to find a plane with at least one point each side
    best_min_points_either_side = 0
    best_i = -1
    best_normal_side_indices = None
    best_opp_side_indices = None

    for hyperplane_indices in hyperplane_indices_to_try:
        normal_side_indices, opp_side_indices = t_split_points(points, hyperplane_indices, normal)
        
        # update if this splitting of points is the most balanced so far
        # also dont want a hyperplane that contains all of the points because that doesnt seperate the points at all 
        #   (numpoints - D-2 taken at start - 1 taken as index 'i')
        min_points_either_side = min(len(normal_side_indices), len(opp_side_indices))
        if min_points_either_side > best_min_points_either_side and min_points_either_side < (num_points-(D-2)-1):
            best_min_points_either_side = min_points_either_side
            best_i = i
            best_normal_side_indices = normal_side_indices
            best_opp_side_indices = opp_side_indices
    
    # checks
    if perform_checks:
        if best_i == -1 or best_min_points_either_side <= 0:
            raise "Something went wrong in triangulation"
    
    # Recursive calls
    normal_side_indices = best_normal_side_indices
    normal_side_indices.extend(range(D-2))
    normal_side_indices.append(best_i)
    normal_side_points = points[:,normal_side_indices]
    simplex_list = t(normal_side_points, simplex_list, D, normal, perform_checks)

    opp_side_indices = best_opp_side_indices
    opp_side_indices.extend(range(D-2))
    opp_side_indices.append(best_i)
    opp_side_points = points[:,opp_side_indices]
    simplex_list = t(opp_side_points, simplex_list, D, normal, perform_checks)

    return simplex_list

def triangulation(points, normal, perform_checks=True):
    """
    shape(points) = (D,m)
    - points is a set of m points in R^D
    points should all be in the D-1 dimensional hyperplane of a simplex
    points need to be not colinear with respect to the D-1 dimensional hyperplane

    returns a list of Simplex objects that partition the D-1 dimensional shape defined by 'points'
    """
    D = points.shape[0]
    return t(points, [], D, normal, perform_checks)

def triangulation_scipy(points, normal, perform_checks=True):
    """
    Use scipy triangulation, my triangulation was bad
    """
    # Our points lie in a plane, with values summing to 1, so can chop off the last dim and add back later
    D, m = points.shape
    scipy_points = points[:D-1].transpose()

    # use scipy triangulation
    tri = scipy.spatial.Delaunay(scipy_points)

    # read out simplices
    # tri.simplices = a (num_simplices,D) shaped matrix of integers, which are indices into tri.points
    # tri.points should be (maybe a permutation) of shape (m,D-1)
    triangulated_simplices = []
    num_simplices, _D = tri.simplices.shape

    if perform_checks:
        # For a D-1 simplex, (we're working in R^D, scipy in R^(D-1)) there are D points in a simplex
        if _D != D:
            print(_D)
            print(D)
            raise "Error using delauny triangulatio with dimensions"
        # For m points a triangulation into D-1 simplices should have m-(D-1) simplices
        # if num_simplices != m-(D-1):
        #     print(m)
        #     print(D)
        #     print(num_simplices)
        #     raise "Unexpected number of simplices in triangulation"

    print(num_simplices)
    
    for i in range(num_simplices):
        simplex_points = np.zeros((D,D))
        for j in range(D):
            simplex_points[:D-1,j] = tri.points[tri.simplices[i,j]]
        # fill in last row using sum of values equals one
        simplex_points[D-1] = 1.0 - np.sum(simplex_points[:D-1], axis=0)
        triangulated_simplices.append(Simplex(simplex_points,perform_checks))
    
    # return
    return triangulated_simplices



if __name__ == "__main__":
    
    # three_d_simplex = UnitSimplex(3)
    # three_d_simplex.compute_triangulation()
    # print("Printing subsimplices for unit 3d simplex:")
    # for sub_simplex in three_d_simplex.triangulation:
    #     print(sub_simplex.points)
    
    # four_d_simplex = UnitSimplex(4)
    # four_d_simplex.compute_triangulation()
    # print("Printing subsimplices for unit 4d simplex:")
    # for sub_simplex in four_d_simplex.triangulation:
    #     print(sub_simplex.points)

    five_d_simplex = UnitSimplex(5)
    five_d_simplex.compute_triangulation()
    print("Expecting 5+0.5*5*4-(5-1)=11 simplices in 5d triangulation, got: {n}".format(n=len(five_d_simplex.triangulation)))
    # for sub_simplex in five_d_simplex.triangulation:
    #     print(sub_simplex.points)

    six_d_simplex = UnitSimplex(6)
    six_d_simplex.compute_triangulation()
    print("Expecting 6+0.5*6*5-(6-1)=16 simplices in 6d triangulation, got: {n}".format(n=len(six_d_simplex.triangulation)))

    seven_d_simplex = UnitSimplex(7)
    seven_d_simplex.compute_triangulation()
    print("Expecting 7+0.5*7*6-(7-1)=22 simplices in 7d triangulation, got: {n}".format(n=len(seven_d_simplex.triangulation)))



    seven_d_simplex = UnitSimplex(8)
    seven_d_simplex.compute_triangulation()
    print("Expecting 8+0.5*8*7-(8-1)=29 simplices in 8d triangulation, got: {n}".format(n=len(seven_d_simplex.triangulation)))

    seven_d_simplex = UnitSimplex(9)
    seven_d_simplex.compute_triangulation()
    print("Expecting 9+0.5*9*8-(7-1)=37 simplices in 9d triangulation, got: {n}".format(n=len(seven_d_simplex.triangulation)))

    seven_d_simplex = UnitSimplex(10)
    seven_d_simplex.compute_triangulation()
    print("Expecting 10+0.5*10*9-(10-1)=46 simplices in 10d triangulation, got: {n}".format(n=len(seven_d_simplex.triangulation)))

    # TODO: 1: Test that these are actually partitions
    # TODO: 2: Write a script that will generate triangulations X times, and use the l1 norms to pick the best one
    # TODO: 3: Write something that will output triangulation in a way that C++ can read it it
    # 0, ..., D-1 = vertex indices
    # D, ..., E writeen as k = i j r (to say that k=r*i+(1-r)*j)
    # each simplex then is just written as a list of vertices
    # triangulation is a list of simplices (1 simplex per line)