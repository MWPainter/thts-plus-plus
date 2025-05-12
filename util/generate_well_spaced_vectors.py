"""
Script to well-spaced points on a surface

Follows the Riesz s-Energy idea from pymoo:
https://pymoo.org/misc/reference_directions.html#Riesz-s-Energy

We have two surfaces of interest:
1. the simplex
    (to generate well spaced weight vectors for evaluation Expected Utility Metric)
2. the (positive) hypersphere
    (x is on positive hypersphere if ||x||=1 and x_i >= 0 for all i)

In general, we have a set of points Z of shape (N,D), where each Z_i is a row vector in R^D. There is a utility/energy 
function which is optimised, and between each optimisation step we ensure that the points remain on the chosen surface 
by a "project_and_clip" function
"""

import csv
import os

import jax
import optax
import jax.numpy as jnp

def project_and_clip_hypersphere(z):
  """
  Projects and clips points z_i onto the positive hypersphere
  """
  z = jnp.clip(z, min=0, max=1)
  return z / jnp.linalg.norm(z,axis=1,keepdims=True)

def project_simplex(z):
  """
  In D dimensions the simplex lies on a D-1 dimensional hyperplane
  This function projects all of the points z_i to the hyperplane that contains the simplex
  N.B. this plane can be defined by points a such that a^T hyperplane_norm - hyperplane_centroid = 0
  """
  dim = z.shape[-1]
  hyperplane_norm = jnp.ones((dim,)) / jnp.sqrt(dim)
  origin_plane_points = z - jnp.matmul(z, hyperplane_norm[:,None]) * hyperplane_norm 
  hyperplane_centroid = jnp.ones((dim,)) / dim
  return hyperplane_centroid + origin_plane_points

def clip_simplex(z):
  """
  Given points that lie on the D-1 dimensional hyperplane that contains the simplex
  This function clips the points z_i to the simplex by taking the point that lies along the line between z_i and the 
  simplex centroid, is in the simplex and is closest to z_i along that line
  """
  dim = z.shape[-1]
  hyperplane_centroid = jnp.ones((dim,)) / dim
  z_to_centroid = hyperplane_centroid - z
  expanded_correction_ratios = jnp.where((z>=0.0), 0.0, -z / z_to_centroid)
  correction_ratios = jnp.max(expanded_correction_ratios,axis=1)
  return jnp.clip(z + correction_ratios[:,None] * z_to_centroid, min=0.0, max=1.0)

def project_and_clip_simplex(z):
  """
  Project and clip points z_i onto the simplex
  """
  return clip_simplex(project_simplex(z))

def distance_squared_matrix(z):
  """
  Computes distance squared matrix, with dist_ij = ||z_i - z_j||^2 
  """
  return jnp.sum((z[:,None,:] - z[None,:,:])**2, axis=2)

def riesz_s_energy_loss(z, s=-1):
  """
  Returns the riesz_s_energy_loss for the points z_i and given s
  s = -1, means that we set s == D^2 following suggestion in 
  https://pymoo.org/misc/reference_directions.html#Riesz-s-Energy
  The +1e-16 is to avoid div by zero and nan gradients
  """
  s = jnp.where(s==-1, z.shape[-1]**2, s)
  distances_sqrd = distance_squared_matrix(z)
  return 0.5 * jnp.sum(jnp.where(distances_sqrd>0, 1/(distances_sqrd**(s/2) + 1e-16), 0))



#####
# Unused old riesz_s_energy_loss
# Keeping for documentation of the jax urgh
#####

def distance_matrix(z):
  """
  Computes a distance matrix, with dist_ij = ||z_i - z_j||
  """
  return jnp.linalg.norm(z[:,None,:] - z[None,:,:], axis=2)

def riesz_s_energy_loss_orig_with_nan_grads(x, s=-1):
  """
  Returns the riesz_s_energy_loss for the points z_i and given s
  s = -1, means that we set s == D^2 following suggestion in 
  https://pymoo.org/misc/reference_directions.html#Riesz-s-Energy

  This function gives nan grads:
  - even though we should avoid div by zero (and we do in forward pass) using the
        jnp.where(distances>0, 1/distances**s, 0)
  - https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
        means that both parts of jnp.where have to have non nan grads, for all input values
  - l2 norm gives nan grads (because d/dx(x^0.5) = 0.5*x^-0.5, which is undef when x=0)
        which means that the distance matrix gives nan grads in the jnp.where
        even though it works in the forward pass

  URGH
  """
  s = jnp.where(s==-1, x.shape[-1]**2, s)
  distances = distance_matrix(x)
  return 0.5 * jnp.sum(jnp.where(distances>0, 1/(distances**s + 1e-16), 0))

#####
# Unused old riesz_s_energy_loss
# Keeping for documentation of the jax urgh
#####



def _generate_well_spaced_points(num_points, dim, project_and_clip_fn, s, opt_lr, opt_iters, prng_key):
  """Jax implementation of 'generate_well_spaced_points'. """
  # Generate free points
  num_free_points = num_points - dim
  free_points = jax.random.uniform(prng_key, shape=(num_free_points,dim), dtype=jnp.float64)
  free_points = project_and_clip_fn(free_points)

  # Default value of s
  s = jnp.where(s==-1, dim**2, s)

  # Log loss fn
  def loss_fn(ps):
    return jnp.log(riesz_s_energy_loss(jnp.concatenate((jnp.eye(dim),ps), axis=0), s=s) + 1e-10)

  # Loss grad fn
  grad_loss_fn = jax.grad(loss_fn)

  # Define optimiser
  optimizer = optax.adam(opt_lr)
  opt_state = optimizer.init(free_points)
  
  # Body for jax.lax.fori_loop
  def loop_fn(iter, loop_state):
    free_points, opt_state = loop_state
    grads = grad_loss_fn(free_points)
    updates, opt_state = optimizer.update(grads, opt_state)
    free_points = optax.apply_updates(free_points, updates)
    free_points = project_and_clip_fn(free_points)
    return (free_points, opt_state)
  
  # Run for 'opt_iter' steps
  loop_state = (free_points, opt_state)
  free_points, opt_state = jax.lax.fori_loop(0, opt_iters, jax.jit(loop_fn), loop_state)
  
  # Return the result (the free points with )
  return jnp.concatenate((jnp.eye(dim),free_points), axis=0)



def generate_well_spaced_points(
    num_points, dim, project_and_clip_fn, s=-1, opt_lr=0.01, opt_iters=5000, prng_key=60415):
  """
  Generates a set of well spaced points. Assumes that the standard basis lie in the desired surface (defined by 
  project_and_clip_fn) and are desired as the first 'dim' points of the well spaced points returned.

  Args:
    num_points: the number of points to generate (num_points > dim)
    dim: the dimension of the points to generate
    project_and_clip_fn: a function that maps (project + clips) points onto the desired surface
    s: the value of s to use for the Riesz s-Energy function
    opt_lr: The learning rate to use for the optimisation
    opt_iters: The number of optimisation iterations to use
    prng_key: a jax prng_key to randomly generate starting points
  """
  jax.config.update("jax_enable_x64", True)
  if not (s > 2 or s == -1):
    raise Exception("The value of s for reisz s-Energy must be set >= 2 or to -1 for default value")
  if isinstance(prng_key, int):
    prng_key = jax.random.PRNGKey(prng_key)
  if num_points <= dim:
    return jnp.eye(dim)[:num_points]
  return _generate_well_spaced_points(num_points, dim, project_and_clip_fn, s, opt_lr, opt_iters, prng_key)



def generate_and_cache_simplex_points(num_points, dim):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  cache_dir_path = os.path.join(dir_path, "cached_simplex_points", str(dim)+"dim")
  cache_file_path = os.path.join(cache_dir_path, str(num_points)+"points.txt")

  if not os.path.exists(cache_dir_path):
    os.makedirs(cache_dir_path)

  generated_points = generate_well_spaced_points(num_points, dim, project_and_clip_simplex)

  with open(cache_file_path, "w+") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    csv_writer.writerows(generated_points)

def generate_and_cache_hypersphere_points(num_points, dim):
  dir_path = os.path.dirname(os.path.realpath(__file__))
  cache_dir_path = os.path.join(dir_path, "cached_hypersphere_points", str(dim)+"dim")
  cache_file_path = os.path.join(cache_dir_path, str(num_points)+"points.txt")

  if not os.path.exists(cache_dir_path):
    os.makedirs(cache_dir_path)

  generated_points = generate_well_spaced_points(num_points, dim, project_and_clip_hypersphere)

  with open(cache_file_path, "w+") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    csv_writer.writerows(generated_points)
  
  





if __name__ == "__main__":
  print("10 well spaced points on 2D simplex:")
  print(generate_well_spaced_points(10,2,project_and_clip_simplex))
  print()

  print("10 well spaced points on 2D hypersphere:")
  print(generate_well_spaced_points(10,2,project_and_clip_hypersphere))
  print()

  print("10 well spaced points on 3D simplex:")
  print(generate_well_spaced_points(10,3,project_and_clip_simplex))
  print()

  print("10 well spaced points on 3D hypersphere:")
  print(generate_well_spaced_points(10,3,project_and_clip_hypersphere))
  print()

  print("100 well spaced points on 3D simplex:")
  thousand_well_spaced_points = generate_well_spaced_points(1000,3,project_and_clip_simplex)
  dists = distance_matrix(thousand_well_spaced_points)
  print(thousand_well_spaced_points)
  dists += jnp.eye(1000) * 1000
  min_dists = jnp.min(dists, axis=1)
  print("Distances to closest point from each z_i")
  print(min_dists)
  print("And min distance between any two points is:")
  print(jnp.min(min_dists))
  print()
  
  # generate_and_cache_simplex_points(10, 2)
  # generate_and_cache_simplex_points(10, 3)
  # generate_and_cache_simplex_points(100, 2)
  # generate_and_cache_simplex_points(100, 3)
  # generate_and_cache_hypersphere_points(10, 2)
  # generate_and_cache_hypersphere_points(10, 3)
  # generate_and_cache_hypersphere_points(100, 2)
  # generate_and_cache_hypersphere_points(100, 3)