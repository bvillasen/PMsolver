using HDF5
using PyPlot
current_dir = pwd()
modules_dir = current_dir * "/modules"
push!(LOAD_PATH, modules_dir)
using tools



const nParticles = 128^3
const totalSteps = 3000
const dt = 2e-10


const Gconst    = 1 #m**2/(kg*s**2)
const mSun = 1     #kg
const pMass = 1

#Domain Parameters
const x_min = 0.0
const y_min = 0.0
const z_min = 0.0
const x_max = 1.0
const y_max = 1.0
const z_max = 1.0
const Lx = x_max - x_min
const Ly = y_max - y_min
const Lz = z_max - z_min

#Grid Properties
const nPoints = 128
const nCells_x = nPoints
const nCells_y = nPoints
const nCells_z = nPoints
const nCells = nCells_x*nCells_y*nCells_z

const dx = Lx / nCells_x
const dy = Ly / nCells_y
const dz = Lz / nCells_z

#Cells positions in grid ( mid-point )
c_pos_x = linspace( x_min + dx/2, x_max - dx/2, nCells_x)
c_pos_y = linspace( y_min + dy/2, y_max - dy/2, nCells_y)
c_pos_z = linspace( z_min + dz/2, z_max - dz/2, nCells_z)

##########################################################################
# Particles Positions
# # Random Uniform
# p_pos_x = (x_max - x_min)*rand( nParticles ) + x_min
# p_pos_y = (y_max - y_min)*rand( nParticles ) + y_min
# p_pos_z = (z_max - z_min)*rand( nParticles ) + z_min
#########################################################################
#Spherically uniform random distribution for initial positions
initialTheta = 2*pi*rand(nParticles)
initialPhi = acos( 2*rand(nParticles) - 1)
R = 0.2
center_x = 0.5
center_y = 0.5
center_z = 0.5
initialR = R^3*rand( nParticles )
initialR = initialR.^(1/3)
p_pos_x = initialR .* cos(initialTheta) .* sin(initialPhi) + center_x
p_pos_y = initialR .* sin(initialTheta) .* sin(initialPhi) + center_y
p_pos_z = initialR .* cos(initialPhi) + center_z

#Particles velocities
p_vel_x = zeros( nParticles )
p_vel_y = zeros( nParticles )
p_vel_z = zeros( nParticles )
##########################################################################
# # Orbiting Particles initial conditions
# D = dx*nCells_x/8
# r = D/2
# v_circ = sqrt( Gconst * r / D^2 )
# p_pos_x = [ 0.5-r 0.5+r ]
# p_pos_y = [ 0.5 0.5 ]
# p_pos_z = [ 0.5-dz/2 0.5-dz/2 ]
# p_vel_x = [ 0. 0. ]
# p_vel_y = [ v_circ -v_circ ]
# p_vel_z = [ 0. 0. ]






dx_all = zeros( nParticles )
dy_all = zeros( nParticles )
dz_all = zeros( nParticles )


# #Cloud-in-Cell density interpolation
# #First get  Nearest Grid Point NGP
# function get_
# idxs_x = floor( Int, (p_pos_x - dx/2)/dx ) + 1
# idxs_y = floor( Int, (p_pos_y - dy/2)/dy ) + 1
# idxs_z = floor( Int, (p_pos_z - dz/2)/dz ) + 1
# inside_x = idxs_x .> x_min
# inside_y = idxs_y .> y_min
# inside_z = idxs_z .> z_min
# p_inside = inside_z .* inside_y .* inside_x
# for i in 1:nParticles
#   dx_all[i] = p_pos_x[i] - c_pos_x[idxs_x[i]]
#   dy_all[i] = p_pos_y[i] - c_pos_y[idxs_y[i]]
#   dz_all[i] = p_pos_z[i] - c_pos_z[idxs_z[i]]
# end







#Array for particles inside the box
p_inside = ones( Bool, nParticles )
function get_particles_outside_CIC( p_inside, p_pos_x, p_pos_y, p_pos_z,
             x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz )
  for i in 1:nParticles
    x, y, z = p_pos_x[i], p_pos_y[i], p_pos_z[i]
    if  ( x<(x_min+2*dx) || x>(x_max-2*dx) )
      p_inside[i] = false
      continue
    end
    if  ( y<(y_min+2*dy) || y>(y_max-2*dy) )
      p_inside[i] = false
      continue
    end
    if  ( z<(z_min+2*dz) || z>(z_max-2*dz) )
      p_inside[i] = false
      continue
    end
  end
end


function get_particles_outside_NPG( p_inside, p_pos_x, p_pos_y, p_pos_z,
             x_min, x_max, y_min, y_max, z_min, z_max)
  for i in 1:nParticles
    x, y, z = p_pos_x[i], p_pos_y[i], p_pos_z[i]
    if  ( x<x_min || x>x_max )
      p_inside[i] = false
      continue
    end
    if  ( y<y_min || y>y_max )
      p_inside[i] = false
      continue
    end
    if  ( z<z_min || z>z_max )
      p_inside[i] = false
      continue
    end
  end
end

outputDir = ""
fileName =  outputDir * "p_pos.h5"
# outFile = h5open( fileName, "w")

function writeSnapshot( n, name, data, outFile; stride=1)
  snapNumber = n < 100 ? "0$(n)" : "$(n)"
  key = name * snapNumber
  pos_x, pos_y, pos_z = data
  outFile[ key * "_pos_x" ] = map( Float32, pos_x[1:stride:end] )
  outFile[ key * "_pos_y" ] = map( Float32, pos_y[1:stride:end] )
  outFile[ key * "_pos_z" ] = map( Float32, pos_z[1:stride:end] )
end

# writeSnapshot( 0, "", [p_pos_x, p_pos_y, p_pos_z], outFile, stride=8 )

function get_indxs_CIC( dx, dy, dz, p_pos_x, p_pos_y, p_pos_z )
  idxs_x = floor( Int, (p_pos_x - 0.5*dx)/dx ) + 1
  idxs_y = floor( Int, (p_pos_y - 0.5*dy)/dy ) + 1
  idxs_z = floor( Int, (p_pos_z - 0.5*dz)/dz ) + 1
  return [ idxs_x, idxs_y, idxs_z ]
end

function get_density_CIC( p_inside, p_pos_x, p_pos_y, p_pos_z, nCells_x, nCells_y, nCells_z, dx, dy, dz, c_pos_x, c_pos_y, c_pos_z )
  idxs_x, idxs_y, idxs_z = get_indxs_CIC( dx, dy, dz, p_pos_x, p_pos_y, p_pos_z )
  #Density array
  rho = zeros( nCells_z, nCells_y, nCells_x )
  for i in 1:nParticles
    if !p_inside[i]
      continue
    end
    idx_x, idx_y, idx_z = idxs_x[i], idxs_y[i], idxs_z[i]
    delta_x = 1 - (p_pos_x[i] - c_pos_x[idx_x])/dx
    delta_y = 1 - (p_pos_y[i] - c_pos_y[idx_y])/dy
    delta_z = 1 - (p_pos_z[i] - c_pos_z[idx_z])/dz
    # println("$(delta_x)    $(delta_y)    $(delta_z)    ")
    rho[idx_z, idx_y, idx_x] += delta_x * delta_y * delta_z
    rho[idx_z, idx_y, idx_x+1] += (1-delta_x) * delta_y * delta_z
    rho[idx_z, idx_y+1, idx_x] += delta_x * (1-delta_y) * delta_z
    rho[idx_z+1, idx_y, idx_x] += delta_x * delta_y * (1-delta_z)
    rho[idx_z+1, idx_y+1, idx_x] += delta_x * (1-delta_y) * (1-delta_z)
    rho[idx_z, idx_y+1, idx_x+1] += (1-delta_x) * (1-delta_y) * delta_z
    rho[idx_z+1, idx_y, idx_x+1] += (1-delta_x) * delta_y * (1-delta_z)
    rho[idx_z+1, idx_y+1, idx_x+1] += (1-delta_x) * (1-delta_y) * (1-delta_z)
  end
  rho /= dx*dy*dz
  return rho, idxs_x, idxs_y, idxs_z
end




function get_indxs_NPG( dx, dy, dz, p_pos_x, p_pos_y, p_pos_z )
  idxs_x = floor( Int, p_pos_x/dx ) + 1
  idxs_y = floor( Int, p_pos_y/dy ) + 1
  idxs_z = floor( Int, p_pos_z/dz ) + 1
  return [ idxs_x, idxs_y, idxs_z ]
end

function get_density_NGP( p_inside, p_pos_x, p_pos_y, p_pos_z, nCells_x, nCells_y, nCells_z, dx, dy, dz )
  idxs_x, idxs_y, idxs_z = get_indxs_NPG( dx, dy, dz, p_pos_x, p_pos_y, p_pos_z )
  #Density array
  rho = zeros( nCells_z, nCells_y, nCells_x )
  for i in 1:nParticles
    if !p_inside[i]
      continue
    end
    idx_x, idx_y, idx_z = idxs_x[i], idxs_y[i], idxs_z[i]
    rho[idx_z, idx_y, idx_x] += 1
  end
  rho /= dx*dy*dz
  return rho, idxs_x, idxs_y, idxs_z
end

function get_potential( density, G, fft_plan_f, fft_plan_b)
  # Apply FFT to rho
  rho_trans = fft_plan_f * (4*pi*Gconst * density)
  rho_trans = G .* rho_trans
  rho_trans[1] = 0
  # Apply inverse FFTW
  phi = real( fft_plan_b * rho_trans )
  return phi
end


function get_grav_force_CIC( phi, dx, dy, dz, i, idx_x, idx_y, idx_z, p_pos_x, p_pos_y, p_pos_z, c_pos_x, c_pos_y, c_pos_z )
  delta_x = 1 - (p_pos_x[i] - c_pos_x[idx_x])/dx
  delta_y = 1 - (p_pos_y[i] - c_pos_y[idx_y])/dy
  delta_z = 1 - (p_pos_z[i] - c_pos_z[idx_z])/dz
  #FOR bottom-left cell
  # X Force component
  phi_l = phi[idx_z, idx_y, idx_x-1]
  phi_r = phi[idx_z, idx_y, idx_x+1]
  g_x_bl = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = phi[idx_z, idx_y-1, idx_x]
  phi_u = phi[idx_z, idx_y+1, idx_x]
  g_y_bl = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = phi[idx_z-1, idx_y, idx_x]
  phi_t = phi[idx_z+1, idx_y, idx_x]
  g_z_bl = -( phi_t - phi_b ) / ( 2*dz )

  #FOR bottom-right
  # X Force component
  phi_l = phi[idx_z, idx_y, idx_x]
  phi_r = phi[idx_z, idx_y, idx_x+2]
  g_x_br = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = phi[idx_z, idx_y-1, idx_x+1]
  phi_u = phi[idx_z, idx_y+1, idx_x+1]
  g_y_br = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = phi[idx_z-1, idx_y, idx_x+1]
  phi_t = phi[idx_z+1, idx_y, idx_x+1]
  g_z_br = -( phi_t - phi_b ) / ( 2*dz )

  #FOR bottom-up
  # X Force component
  phi_l = phi[idx_z, idx_y+1, idx_x-1]
  phi_r = phi[idx_z, idx_y+1, idx_x+1]
  g_x_bu = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = phi[idx_z, idx_y, idx_x]
  phi_u = phi[idx_z, idx_y+2, idx_x]
  g_y_bu = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = phi[idx_z-1, idx_y+1, idx_x]
  phi_t = phi[idx_z+1, idx_y+1, idx_x]
  g_z_bu = -( phi_t - phi_b ) / ( 2*dz )

  #FOR bottom-right-up
  # X Force component
  phi_l = phi[idx_z, idx_y+1, idx_x]
  phi_r = phi[idx_z, idx_y+1, idx_x+2]
  g_x_bru = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = phi[idx_z, idx_y, idx_x+1]
  phi_u = phi[idx_z, idx_y+2, idx_x+1]
  g_y_bru = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = phi[idx_z-1, idx_y+1, idx_x+1]
  phi_t = phi[idx_z+1, idx_y+1, idx_x+1]
  g_z_bru = -( phi_t - phi_b ) / ( 2*dz )

  #FOR top-left
  # X Force component
  phi_l = phi[idx_z+1, idx_y, idx_x-1]
  phi_r = phi[idx_z+1, idx_y, idx_x+1]
  g_x_tl = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = phi[idx_z+1, idx_y-1, idx_x]
  phi_u = phi[idx_z+1, idx_y+1, idx_x]
  g_y_tl = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = phi[idx_z, idx_y, idx_x]
  phi_t = phi[idx_z+2, idx_y, idx_x]
  g_z_tl = -( phi_t - phi_b ) / ( 2*dz )

  #FOR top-right
  # X Force component
  phi_l = phi[idx_z+1, idx_y, idx_x]
  phi_r = phi[idx_z+1, idx_y, idx_x+2]
  g_x_tr = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = phi[idx_z+1, idx_y-1, idx_x+1]
  phi_u = phi[idx_z+1, idx_y+1, idx_x+1]
  g_y_tr = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = phi[idx_z, idx_y, idx_x+1]
  phi_t = phi[idx_z+2, idx_y, idx_x+1]
  g_z_tr = -( phi_t - phi_b ) / ( 2*dz )

  #FOR top-up
  # X Force component
  phi_l = phi[idx_z+1, idx_y+1, idx_x-1]
  phi_r = phi[idx_z+1, idx_y+1, idx_x+1]
  g_x_tu = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = phi[idx_z+1, idx_y, idx_x]
  phi_u = phi[idx_z+1, idx_y+2, idx_x]
  g_y_tu = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = phi[idx_z, idx_y+1, idx_x]
  phi_t = phi[idx_z+2, idx_y+1, idx_x]
  g_z_tu = -( phi_t - phi_b ) / ( 2*dz )

  #FOR top-right-up
  # X Force component
  phi_l = phi[idx_z+1, idx_y+1, idx_x]
  phi_r = phi[idx_z+1, idx_y+1, idx_x+2]
  g_x_tru = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = phi[idx_z+1, idx_y, idx_x+1]
  phi_u = phi[idx_z+1, idx_y+2, idx_x+1]
  g_y_tru = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = phi[idx_z, idx_y+1, idx_x+1]
  phi_t = phi[idx_z+2, idx_y+1, idx_x+1]
  g_z_tru = -( phi_t - phi_b ) / ( 2*dz )

  g_x = g_x_bl*delta_x*delta_y*delta_z         + g_x_br*(1-delta_x)*delta_y*delta_z +
        g_x_bu*delta_x*(1-delta_y)*delta_z     + g_x_bru*(1-delta_x)*(1-delta_y)*delta_z +
        g_x_tl*delta_x*delta_y*(1-delta_z)     + g_x_tr*(1-delta_x)*delta_y*(1-delta_z) +
        g_x_tu*delta_x*(1-delta_y)*(1-delta_z) + g_x_tru*(1-delta_x)*(1-delta_y)*(1-delta_z)

  g_y = g_y_bl*delta_x*delta_y*delta_z         + g_y_br*(1-delta_x)*delta_y*delta_z +
        g_y_bu*delta_x*(1-delta_y)*delta_z     + g_y_bru*(1-delta_x)*(1-delta_y)*delta_z +
        g_y_tl*delta_x*delta_y*(1-delta_z)     + g_y_tr*(1-delta_x)*delta_y*(1-delta_z) +
        g_y_tu*delta_x*(1-delta_y)*(1-delta_z) + g_y_tru*(1-delta_x)*(1-delta_y)*(1-delta_z)

  g_z = g_z_bl*delta_x*delta_y*delta_z         + g_z_br*(1-delta_x)*delta_y*delta_z +
        g_z_bu*delta_x*(1-delta_y)*delta_z     + g_z_bru*(1-delta_x)*(1-delta_y)*delta_z +
        g_z_tl*delta_x*delta_y*(1-delta_z)     + g_z_tr*(1-delta_x)*delta_y*(1-delta_z) +
        g_z_tu*delta_x*(1-delta_y)*(1-delta_z) + g_z_tru*(1-delta_x)*(1-delta_y)*(1-delta_z)

  return [ g_x g_y g_z ]
end

function get_grav_force_NGP( phi, idx_x, idx_y, idx_z )
  # X Force component
  phi_l = idx_x > 1 ? phi[idx_z, idx_y, idx_x-1] : phi[idx_z, idx_y, end ]
  phi_r = idx_x < nCells_x ? phi[idx_z, idx_y, idx_x+1] : phi[idx_z, idx_y, 1 ]
  g_x = -( phi_r - phi_l ) / ( 2*dx )
  # Y Force component
  phi_d = idx_y > 1 ? phi[idx_z, idx_y-1, idx_x] : phi[idx_z, end, idx_x ]
  phi_u = idx_y < nCells_y ? phi[idx_z, idx_y+1, idx_x] : phi[idx_z, 1, idx_x ]
  g_y = -( phi_u - phi_d ) / ( 2*dy )
  # Z Force component
  phi_b = idx_z > 1 ? phi[idx_z-1, idx_y, idx_x] : phi[end, idx_y, idx_x ]
  phi_t = idx_z < nCells_z ? phi[idx_z+1, idx_y, idx_x] : phi[1, idx_y, idx_x ]
  g_z = -( phi_t - phi_b ) / ( 2*dz )
  return [ g_x g_y g_z ]
end

function update_leapfrog( i, dt, p_pos_x, p_pos_y, p_pos_z, p_vel_x, p_vel_y, p_vel_z, g_x, g_y, g_z )
  #Update velocities using current forces
  p_vel_x[i] = p_vel_x[i] + dt*g_x
  p_vel_y[i] = p_vel_y[i] + dt*g_y
  p_vel_z[i] = p_vel_z[i] + dt*g_z
  #Update positions using updated velocities
  p_pos_x[i] = p_pos_x[i] + dt*p_vel_x[i]
  p_pos_y[i] = p_pos_y[i] + dt*p_vel_y[i]
  p_pos_z[i] = p_pos_z[i] + dt*p_vel_z[i]
end

function initialize_FFT( nCells_x, nCells_y, nCells_z, Lx, Ly, Lz, dx, dy, dz )
  dens = zeros( nCells_z, nCells_y, nCells_x )
  # Initialize FFTW
  FFTW.set_num_threads(8)
  fft_plan_fwd  = plan_fft( dens, flags=FFTW.ESTIMATE, timelimit=20 )
  fft_plan_bkwd = plan_ifft( dens, flags=FFTW.ESTIMATE, timelimit=20 )

  fft_kx = sin( pi/Lx*dx * linspace(0, nCells_x-1, nCells_x) ).^2
  fft_ky = sin( pi/Ly*dy * linspace(0, nCells_y-1, nCells_y) ).^2
  fft_kz = sin( pi/Lz*dz * linspace(0, nCells_z-1, nCells_z) ).^2
  G = zeros( nCells_z, nCells_y, nCells_x )
  for i in 1:nCells_x
    sin_kx = fft_kx[i]
    for j in 1:nCells_y
      sin_ky = fft_ky[j]
      for k in 1:nCells_z
        sin_kz =  fft_kz[k]
        G[k,j,i] = -1/ ( sin_kx + sin_ky + sin_kz )
      end
    end
  end
  return [ G, fft_plan_fwd, fft_plan_bkwd ]
end

function update_particles_CIC( dt, dx, dy, dz, phi, idxs_x, idxs_y, idxs_z, p_inside,
                               p_pos_x, p_pos_y, p_pos_z,
                               p_vel_x, p_vel_y, p_vel_z,
                               c_pos_x, c_pos_y, c_pos_z  )
  for i in 1:nParticles
    if !p_inside[i]
      continue
    end
    idx_x, idx_y, idx_z = idxs_x[i], idxs_y[i], idxs_z[i]
    g_x, g_y, g_z = get_grav_force_CIC( phi, dx, dy, dz, i, idx_x, idx_y, idx_z, p_pos_x, p_pos_y, p_pos_z, c_pos_x, c_pos_y, c_pos_z )
    update_leapfrog( i, dt, p_pos_x, p_pos_y, p_pos_z, p_vel_x, p_vel_y, p_vel_z, g_x, g_y, g_z )
  end
end




function update_particles_NGP( dt, phi, idxs_x, idxs_y, idxs_z, p_inside,
                               p_pos_x, p_pos_y, p_pos_z,
                               p_vel_x, p_vel_y, p_vel_z,  )
  for i in 1:nParticles
    if !p_inside[i]
      continue
    end
    idx_x, idx_y, idx_z = idxs_x[i], idxs_y[i], idxs_z[i]
    g_x, g_y, g_z = get_grav_force_NGP( phi, idx_x, idx_y, idx_z )
    update_leapfrog( i, dt, p_pos_x, p_pos_y, p_pos_z, p_vel_x, p_vel_y, p_vel_z, g_x, g_y, g_z )
  end
end



function plot_slides( nStep, cut, rho, phi )
  img = log( 1e-7*rho[cut,:,:] + 1 )
  clf()
  imshow(img)
  colorbar()
  title( "Density" )
  savefig("images/density_$(nStep/2 -1 ).png")
  img = phi[cut,:,:]
  clf()
  imshow(img)
  colorbar()
  title( "Potential" )
  savefig("images/potential_$(nStep/2 -1 ).png")
end

function advance_step_CIC( nStep, nCells_x, nCells_y, nCells_z, dx, dy, dz, p_inside,
                           p_pos_x, p_pos_y, p_pos_z, p_vel_x, p_vel_y, p_vel_z,
                           c_pos_x, c_pos_y, c_pos_z,
                           G, fft_plan_fwd, fft_plan_bkwd )
  get_particles_outside_CIC( p_inside, p_pos_x, p_pos_y, p_pos_z, x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz)
  rho, idxs_x, idxs_y, idxs_z = get_density_CIC( p_inside, p_pos_x, p_pos_y, p_pos_z, nCells_x, nCells_y, nCells_z, dx, dy, dz, c_pos_x, c_pos_y, c_pos_z )
  phi = get_potential( rho, G, fft_plan_fwd, fft_plan_bkwd)
  update_particles_CIC( dt,  dx, dy, dz, phi, idxs_x, idxs_y, idxs_z, p_inside, p_pos_x, p_pos_y, p_pos_z, p_vel_x, p_vel_y, p_vel_z, c_pos_x, c_pos_y, c_pos_z  )
  if ( mod( nStep, 2  ) == 0 )
    plot_slides( nStep, Int(nCells_z/2), rho, phi )
  end
end



function advance_step_NGP( nStep, nCells_x, nCells_y, nCells_z, dx, dy, dz, p_inside,
                           p_pos_x, p_pos_y, p_pos_z, p_vel_x, p_vel_y, p_vel_z,
                           G, fft_plan_fwd, fft_plan_bkwd )
  get_particles_outside_NPG( p_inside, p_pos_x, p_pos_y, p_pos_z, x_min, x_max, y_min, y_max, z_min, z_max)
  rho, idxs_x, idxs_y, idxs_z = get_density_NGP( p_inside, p_pos_x, p_pos_y, p_pos_z, nCells_x, nCells_y, nCells_z, dx, dy, dz )
  phi = get_potential( rho, G, fft_plan_fwd, fft_plan_bkwd)
  update_particles_NGP( dt, phi, idxs_x, idxs_y, idxs_z, p_inside, p_pos_x, p_pos_y, p_pos_z, p_vel_x, p_vel_y, p_vel_z,  )
  if ( mod( nStep, 2  ) == 0 )
    plot_slides( nStep, Int(nCells_z/2), rho, phi )
  end

end


###################################################################################
G, fft_plan_fwd, fft_plan_bkwd = initialize_FFT( nCells_x, nCells_y, nCells_z, Lx, Ly, Lz, dx, dy, dz )

# First half-step for leapfrog method
get_particles_outside_CIC( p_inside, p_pos_x, p_pos_y, p_pos_z, x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz)
# rho, idxs_x, idxs_y, idxs_z = get_density_NGP( p_inside, p_pos_x, p_pos_y, p_pos_z, nCells_x, nCells_y, nCells_z, dx, dy, dz )
rho, idxs_x, idxs_y, idxs_z = get_density_CIC( p_inside, p_pos_x, p_pos_y, p_pos_z, nCells_x, nCells_y, nCells_z, dx, dy, dz, c_pos_x, c_pos_y, c_pos_z )
phi = get_potential( rho, G, fft_plan_fwd, fft_plan_bkwd)
for i in 1:nParticles
  if !p_inside[i]
    continue
  end
  idx_x, idx_y, idx_z = idxs_x[i], idxs_y[i], idxs_z[i]
  g_x, g_y, g_z = get_grav_force_NGP( phi, idx_x, idx_y, idx_z )
  p_vel_x[i] = p_vel_x[i] + 0.5*dt*g_x
  p_vel_y[i] = p_vel_y[i] + 0.5*dt*g_y
  p_vel_z[i] = p_vel_z[i] + 0.5*dt*g_z
end
##################################################################################
# Start simulation
tic()
time_total = 0
stepsPerWrite = 10
for nStep in 1:totalSteps
  printProgress( nStep-1, totalSteps, time_total)
  time_step = @elapsed advance_step_CIC( nStep, nCells_x, nCells_y, nCells_z, dx, dy, dz, p_inside, p_pos_x, p_pos_y, p_pos_z, p_vel_x, p_vel_y, p_vel_z, c_pos_x, c_pos_y, c_pos_z, G, fft_plan_fwd, fft_plan_bkwd )
  time_total += time_step
  # if ( mod( nStep, stepsPerWrite  ) == 0 )
  #   writeSnapshot( nStep, "", [p_pos_x, p_pos_y, p_pos_z], outFile, stride=8 )
  # end
end
printProgress( totalSteps, totalSteps, time_total)
println("\nTotal Time: $(time_total) secs")
toc()

# close( outFile )
