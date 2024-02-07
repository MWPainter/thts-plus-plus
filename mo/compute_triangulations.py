import numpy as np 
import scipy
from collections import defaultdict
from io import StringIO


class StringBuilder(object):
    """Python has a lack of good string building objects. So here's a custom
    StringBuilder object.

    Attributes:
        _arr: An internal array object that we're using for the string building.
        _is_empty: Boolean keeping track of if the array is empty or not.
    """
    def __init__(self):
        self._str = StringIO()
        self._is_empty = True

    def append(self, string, with_line_break=False):
        """Appends a string in the String Builder.
        Args:
            string: The string to be appended
            with_line_break: If True then a '\n' is appended to the end of
            the string
        """
        self._is_empty = (self._is_empty and string == '')
        self._str.write(string)
        if with_line_break:
            self._str.write('\n')

    def is_empty(self):
        """Returns if the string builder is currently empty."""
        return self._is_empty

    def to_string(self):
        """Return's the built string in the string builder."""
        return self._str.getvalue()
    



class Hyperplane:
    """
    Forms a hyperplane from a set of points
    Assumes that 
    """
    def __init__(self, points, perform_checks=False):
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
    def __init__(self, points, perform_checks=False, random_joggle=False):
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
        self.compute_edge_points(random_joggling=random_joggle)
        self.compute_vertex_labels()
    
    def compute_edge_points(self, random_joggling=False):
        """
        Computes 0.5*(u+v) for each u,v in the simplex

        Non random joggling fails on dim=8, so I think this case generates colinear points and makes the 
        triangulation fail?

        Eitherway, non random joggling fails for some reason right now, so going to use random joggling
        """
        num_edge_points = int(0.5 * self.D * (self.D - 1))
        self.edge_points = np.zeros((self.D, num_edge_points))
        self.edge_point_to_edge = {}
        self.vertex_to_edge_points = defaultdict(list)
        self.ratios = np.linspace(0.4, 0.6, num=num_edge_points)
        if random_joggling:
            self.ratios = 0.4 + 0.2 * np.random.rand(num_edge_points)
        i = 0
        for j in range(self.D):
            for k in range(j+1, self.D):
                self.edge_points[:,i] = self.ratios[i] * self.points[:,j] + (1.0-self.ratios[i]) * self.points[:,k]
                self.edge_point_to_edge[tuple(self.edge_points[:,i])] = (j,k)
                self.vertex_to_edge_points[j].append(self.edge_points[:,i])
                self.vertex_to_edge_points[k].append(self.edge_points[:,i])
                i += 1

    def compute_vertex_labels(self):
        self.vertex_labels = {}
        for i in range(self.D):
            self.vertex_labels[tuple(self.points[:,i])] = i
        num_edge_points = self.edge_points.shape[1]
        for i in range(num_edge_points):
            self.vertex_labels[tuple(self.edge_points[:,i])] = self.D + i

    def compute_max_linf_norm_ratio(self, base_max_l1_norm=2.0):
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
                l1_norm = np.linalg.norm(self.points[:,j]-self.points[:,k], ord=np.inf)
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
    
    def contains_point_in_simplex(self, point):
        hyperplanes = []
        for i in range(self.D):
            face_points = np.zeros((self.D, self.D))
            face_points[:,:i] = self.points[:,:i] 
            face_points[:,i:self.D-1] = self.points[:,i+1:]
            face_points[:,self.D-1] = self.points[:,0] + self.normal
            hyperplanes.append(Hyperplane(face_points, perform_checks=self.perform_checks))
        
        centroid = np.mean(self.points, axis=0)

        for i in range(self.D):
            centroid_normal_side = hyperplanes[i].point_is_normal_side(centroid)
            point_normal_side = hyperplanes[i].point_is_normal_side(point)
            if centroid_normal_side != point_normal_side:
                return False
            
        return True
    
    def triangulation_to_string(self):
        sb = StringBuilder()

        # add header info
        num_edge_points = self.edge_points.shape[1]
        sb.append("{v}\n".format(v=self.D+num_edge_points))
        sb.append("{s}\n".format(s=len(self.triangulation)))

        # add simplex points
        for i in range(self.D):
            sb.append("{i}\n".format(i=i))

        # add edge points
        i = 0
        for j in range(self.D):
            for k in range(j+1, self.D):
                lab = i + self.D
                sb.append("{i} {j} {k} {r}\n".format(i=lab,j=j,k=k,r=self.ratios[i]))
                i += 1

        # add triangulations
        for i, triangle in enumerate(self.triangulation):
            for j in range(self.D):
                triangle_point = triangle.points[:,j]
                label = self.vertex_labels[tuple(triangle_point)]
                sb.append(str(label))
                if (j < self.D-1):
                    sb.append(" ")
            sb.append("\n")

        return sb.to_string()
    
    def write_triangulation_to_file(self, filename):
        with open(filename, "w") as f:
            f.write(self.triangulation_to_string())

            
                



class UnitSimplex(Simplex):
    """
    Helper to make unit simplex
    The identity matrix in D dimensions forms a D-1-simplex
    """
    def __init__(self, D, random_joggle=False):
        super().__init__(np.identity(D), random_joggle=random_joggle)



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
    best_hyperplane_indices = None
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
            best_hyperplane_indices = hyperplane_indices
            best_normal_side_indices = normal_side_indices
            best_opp_side_indices = opp_side_indices
    
    # checks
    if perform_checks:
        if best_min_points_either_side <= 0:
            raise "Something went wrong in triangulation"
    
    # Recursive calls
    normal_side_indices = best_normal_side_indices
    normal_side_indices.extend(best_hyperplane_indices)
    normal_side_points = points[:,normal_side_indices]
    simplex_list = t(normal_side_points, simplex_list, D, normal, perform_checks)

    opp_side_indices = best_opp_side_indices
    opp_side_indices.extend(best_hyperplane_indices)
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

def generate_point_in_simplex(dim):
    rand_exp = np.random.exponential(1.0, dim)
    sum_exp = np.sum(rand_exp)
    ordered_vals_in_unit_interval = np.array([np.sum(rand_exp[:i])/sum_exp for i in range(0,dim+1)])
    return ordered_vals_in_unit_interval[1:] - ordered_vals_in_unit_interval[:dim]



if __name__ == "__main__":
    
    ##########
    # TESTING #1
    # Generate triangulations
    # Check expected number of simplices in triangulation
    # CVheck that the triangulations are a complete partition of the simplex
    # (i.e. we triangulate a unit simplex, and then check that the triangulation is a partition of the unit simplex)
    ##########

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

    # five_d_simplex = UnitSimplex(5)
    # five_d_simplex.compute_triangulation()
    # print("Expecting 5+0.5*5*4-(5-1)=11 simplices in 5d triangulation, got: {n}".format(n=len(five_d_simplex.triangulation)))
    # # for sub_simplex in five_d_simplex.triangulation:
    # #     print(sub_simplex.points)

    # print("Testing 5d simplex (if no fail output then passed)")
    # for i in range(10000):
    #     simplex_point = generate_point_in_simplex(5)
    #     found_triangle_containing_point = False
    #     for triangle in five_d_simplex.triangulation:
    #         if triangle.contains_point_in_simplex(simplex_point):
    #             found_triangle_containing_point = True
    #             break
    #     if not found_triangle_containing_point:
    #         print("Failed test with point:")
    #         print(simplex_point)
    #         break

    # six_d_simplex = UnitSimplex(6)
    # six_d_simplex.compute_triangulation()
    # print("Expecting 6+0.5*6*5-(6-1)=16 simplices in 6d triangulation, got: {n}".format(n=len(six_d_simplex.triangulation)))

    # print("Testing 6d simplex (if no fail output then passed)")
    # for i in range(10000):
    #     simplex_point = generate_point_in_simplex(6)
    #     found_triangle_containing_point = False
    #     for triangle in six_d_simplex.triangulation:
    #         if triangle.contains_point_in_simplex(simplex_point):
    #             found_triangle_containing_point = True
    #             break
    #     if not found_triangle_containing_point:
    #         print("Failed test with point:")
    #         print(simplex_point)
    #         break

    # seven_d_simplex = UnitSimplex(7)
    # seven_d_simplex.compute_triangulation()
    # print("Expecting 7+0.5*7*6-(7-1)=22 simplices in 7d triangulation, got: {n}".format(n=len(seven_d_simplex.triangulation)))

    # print("Testing 7d simplex (if no fail output then passed)")
    # for i in range(10000):
    #     simplex_point = generate_point_in_simplex(7)
    #     found_triangle_containing_point = False
    #     for triangle in seven_d_simplex.triangulation:
    #         if triangle.contains_point_in_simplex(simplex_point):
    #             found_triangle_containing_point = True
    #             break
    #     if not found_triangle_containing_point:
    #         print("Failed test with point:")
    #         print(simplex_point)
    #         break

    # simplex = UnitSimplex(8)
    # simplex.compute_triangulation()
    # print("Expecting 8+0.5*8*7-(8-1)=29 simplices in 8d triangulation, got: {n}".format(n=len(simplex.triangulation)))

    # print("Testing 8d simplex (if no fail output then passed)")
    # for i in range(10000):
    #     simplex_point = generate_point_in_simplex(8)
    #     found_triangle_containing_point = False
    #     for triangle in simplex.triangulation:
    #         if triangle.contains_point_in_simplex(simplex_point):
    #             found_triangle_containing_point = True
    #             break
    #     if not found_triangle_containing_point:
    #         print("Failed test with point:")
    #         print(simplex_point)
    #         break

    # simplex = UnitSimplex(9)
    # simplex.compute_triangulation()
    # print("Expecting 9+0.5*9*8-(7-1)=37 simplices in 9d triangulation, got: {n}".format(n=len(simplex.triangulation)))

    # print("Testing 9d simplex (if no fail output then passed)")
    # for i in range(10000):
    #     simplex_point = generate_point_in_simplex(9)
    #     found_triangle_containing_point = False
    #     for triangle in simplex.triangulation:
    #         if triangle.contains_point_in_simplex(simplex_point):
    #             found_triangle_containing_point = True
    #             break
    #     if not found_triangle_containing_point:
    #         print("Failed test with point:")
    #         print(simplex_point)
    #         break

    # simplex = UnitSimplex(10)
    # simplex.compute_triangulation()
    # print("Expecting 10+0.5*10*9-(10-1)=46 simplices in 10d triangulation, got: {n}".format(n=len(simplex.triangulation)))

    # print("Testing 6d simplex (if no fail output then passed)")
    # for i in range(10000):
    #     simplex_point = generate_point_in_simplex(10)
    #     found_triangle_containing_point = False
    #     for triangle in simplex.triangulation:
    #         if triangle.contains_point_in_simplex(simplex_point):
    #             found_triangle_containing_point = True
    #             break
    #     if not found_triangle_containing_point:
    #         print("Failed test with point:")
    #         print(simplex_point)
    #         break
    
    ##########
    # TESTING #2
    # Generate triangulations
    # Output l1 norm ratios
    ##########

    # for i in range(25):
    #     simplex = UnitSimplex(10)
    #     try:
    #         simplex.compute_triangulation()
    #         print(i)
    #         max_linf_norm_ratio = 0.0
    #         for triangle in simplex.triangulation:
    #             linf_norm_ratio = triangle.compute_max_linf_norm_ratio()
    #             print(linf_norm_ratio)
    #             if linf_norm_ratio > max_linf_norm_ratio:
    #                 max_linf_norm_ratio = linf_norm_ratio
    #         print(max_linf_norm_ratio)
    #         print()
    #     except:
    #         print(i)
    #         print("errored")
    #         print()
    
    ##########
    # TESTING #3
    # Print triangulation file string to stdout
    ##########

    # simplex = UnitSimplex(5)
    # simplex.compute_triangulation()
    # print("Example 5d triangulation output:")
    # print(simplex.triangulation_to_string())
    # print()
    # print()

    # simplex = UnitSimplex(9)
    # simplex.compute_triangulation()
    # print("Example 9d triangulation output:")
    # print(simplex.triangulation_to_string())
    # print()
    # print()
    
    ##########
    # Printing out the real things
    # I'm going to arbitrarily go up to 25, because probably more than what I'm going to use
    # Unsure why the linear joggling didnt work for 13 / 18 / 23
    # Tried 28 out of curiosity and that works :S
    ##########

    # NOTE: the 2d and 3d cases we hand wrote
    for i in range(4,26):
        random_joggle = False
        if i in [13,18,23]:
            random_joggle = True
        simplex = UnitSimplex(i, random_joggle=random_joggle)
        simplex.compute_triangulation()
        simplex.write_triangulation_to_file(".cache/{i}_triangulation.txt".format(i=i))