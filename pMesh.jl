using HDF5
current_dir = pwd()
modules_dir = current_dir * "/modules"
push!(LOAD_PATH, modules_dir)
using tools



const nParticles = 128^3
const totalSteps = 1000


G    = 1 #m**2/(kg*s**2)
mSun = 1     #kg
pMass = 1
initialR =  500

#Domain Parameters
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
z_min, z_max = 0.0, 1.0
Lx = x_max - x_min
Ly = y_max - y_min
Lz = z_max - z_min

#Grid Properties
const nPoints = 128
const nCells_x = nPoints
const nCells_y = nPoints
const nCells_z = nPoints
const nCells = nCells_x*nCells_y*nCells_z

dx = Lx / nCells_x
dy = Ly / nCells_y
dz = Lz / nCells_z

#Cells positions in grid ( mid-point )
c_pos_x = linspace( x_min + dx/2, x_max - dx/2, nCells_x)
c_pos_y = linspace( y_min + dy/2, y_max - dy/2, nCells_y)
c_pos_z = linspace( z_min + dz/2, z_max - dz/2, nCells_z)

#Particles Positions
p_pos_x = (x_max - x_min)*rand( nParticles ) + x_min
p_pos_y = (y_max - y_min)*rand( nParticles ) + y_min
p_pos_z = (z_max - z_min)*rand( nParticles ) + z_min

#Particles velocities
p_vel_x = zeros( nParticles )
p_vel_y = zeros( nParticles )
p_vel_z = zeros( nParticles )

#Density array
rho = zeros( nCells_z, nCells_y, nCells_x )

#Get Density Nearest Grid Point NGP
idxs_x = floor( Int, p_pos_x/dx ) + 1
idxs_y = floor( Int, p_pos_y/dy ) + 1
idxs_z = floor( Int, p_pos_z/dz ) + 1
for i in 1:nParticles
  idx_x, idx_y, idx_z = idxs_x[i], idxs_y[i], idxs_z[i]
  rho[idx_z, idx_y, idx_x] += 1
end
rho *= dx*dy*dz

# Get the potential from Density
fft_kx = 2*pi/Lx * linspace(0, nCells_x-1, nCells_x)
fft_ky = 2*pi/Ly * linspace(0, nCells_y-1, nCells_y)
fft_kz = 2*pi/Lz * linspace(0, nCells_z-1, nCells_z)

# fft_kx = sin( pi/Lx * linspace(0, nCells_x-1, nCells_x) ).^2
# fft_ky = sin( pi/Ly * linspace(0, nCells_y-1, nCells_y) ).^2
# fft_kz = sin( pi/Lz * linspace(0, nCells_z-1, nCells_z) ).^2


FFTW.set_num_threads(8)
fft_plan_fwd = plan_fft( rho, flags=FFTW.ESTIMATE, timelimit=20 )
# Apply FFT to rho
rho_trans = fft_plan_fwd * rho
tic()
for k = 1:100
  rho_trans = fft_plan_fwd * rho
end
time = toc()

tic()
for i in 1:nCells_x
  sin_kx = fft_kx[i]
  for j in 1:nCells_y
    sin_ky = fft_ky[j]
    for k in 1:nCells_z
      if i==1 && j==1 && k == 1
        rho_trans[1,1,1] = 0
        continue
      end
      # kx, ky, kz = fft_kx[i], fft_ky[j], fft_kz[k]
      # G = -1/ ( sin(kx/2)^2 + sin(ky/2)^2 + sin(kz/2)^2 )
      sin_kz =  fft_kz[k]
      G = -1/ ( sin_kx + sin_ky + sin_kz )
      rho_trans[k, j, i] *= G
    end
  end
end
time=toc()
